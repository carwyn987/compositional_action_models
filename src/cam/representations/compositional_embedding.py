"""Learnable compositional embeddings of PDDL operators, trained end to end with the policy.

The operator travels in the observation as a fixed-size *structure array* (StructuredOperatorEncoder,
environment side). A features extractor in the policy (CompositionalOperatorExtractor) unpacks it into
components and composes their learnable embeddings into one operator embedding of size
--operator-embedding-dim, appended to the other observation parts exactly like every other embedding
condition (docs/policy_inputs.md). Being part of the policy, every embedding and function below is
trained by the RL losses.

Components (each embedded in R^component_dim):
    type            every object type (e.g. block) has an embedding
    variable        every parameter *position* has an embedding: variables are identified by position,
                    not by name, so renaming ?o to ?x changes nothing and variable i lines up with
                    grounding slot i (docs/policy_inputs.md)
    typed variable  TYPED(variable, type): the type acts on the variable through a learned function
    predicate       every predicate the environment can evaluate (e.g. on-table) has an embedding
    literal         LITERAL(predicate, argument_1, ..., argument_A): the predicate applied to its
                    typed-variable arguments in argument order, so (on ?a ?b) differs from (on ?b ?a)
    name            the operator name's fixed text embedding (--text-backend), through a learned map

Architectures, i.e. how the components are composed (--compositional-architecture):
    tree       TreeComposition: learned functions following the PDDL syntax
                   NOT(literal)                       negated preconditions and delete effects
                   AND({literals})                    conjunction: a DeepSets sum, order-invariant
                   PRE(AND(preconditions)), EFF(AND(effects)), PARAMETERS({typed variables})
                   OPERATOR(NAME, PARAMETERS, PRE, EFF) -> the operator embedding
    slots      SlotComposition: no hand-written composition. Tokens for the name, typed variables and
               literals (each plus an embedding of its kind) are grouped by slot attention
               (Locatello et al., 2020) into learned slots, flattened into the operator embedding
    geometric  GeometricComposition (geometric_embedding.py): the components laid out on a grid, read
               by a CNN and slot attention

Structure array (ids are stored as floats; 0 means empty):
    [type id per parameter (P)] [per literal: section, predicate id, argument refs (L x (2 + A))]
    [name text embedding (D)]
with P = max_parameters, L = max_literals, A = the largest predicate arity, D = name_dim. An argument
ref is the parameter position + 1. Literals are sorted by (section, predicate, arguments), so the array
is canonical: renaming variables or reordering literals in the PDDL source gives the same array.
"""

from dataclasses import dataclass
from typing import NamedTuple

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from cam.domain.pddl.pddl import PDDLOperator
from cam.domain.symbols import Predicate
from cam.policies.feature_extractors import OPERATOR_EMBEDDING_KEY
from cam.representations.operator_encoder import OperatorEncoder
from cam.representations.text_backends import TextEmbeddingBackend

# Literal sections, as stored in the structure array.
PRECONDITION, NEGATED_PRECONDITION, ADD, DELETE = 1, 2, 3, 4
# Component kinds (used by the slots and geometric architectures): a literal's kind is its section + 1.
NAME_KIND, PARAMETER_KIND = 0, 1
NUM_KINDS = 6


@dataclass(frozen=True)
class OperatorLayout:
    """Sizes and vocabularies of the structure array, shared by StructuredOperatorEncoder and the
    architectures so both sides agree. Predicate and type ids are their index + 1 (0 = empty)."""

    predicates: tuple[str, ...]
    types: tuple[str, ...]
    max_predicate_arity: int
    max_parameters: int
    max_literals: int
    name_dim: int = 0

    @classmethod
    def from_predicate_arities(
        cls, predicate_arities: dict[str, int], types: list[str], max_parameters: int, max_literals: int,
        name_dim: int = 0,
    ) -> "OperatorLayout":
        return cls(
            predicates=tuple(sorted(predicate_arities)),
            types=tuple(types),
            max_predicate_arity=max(predicate_arities.values(), default=0),
            max_parameters=max_parameters,
            max_literals=max_literals,
            name_dim=name_dim,
        )

    @property
    def literal_width(self) -> int:
        return 2 + self.max_predicate_arity

    @property
    def size(self) -> int:
        return self.max_parameters + self.max_literals * self.literal_width + self.name_dim


class StructuredOperatorEncoder(OperatorEncoder):
    """Environment side: an operator -> its structure array (module docstring), which
    CompositionalOperatorExtractor decodes in the policy.

    name_backend embeds the operator name (needed exactly when layout.name_dim > 0). Raises ValueError
    for an operator the layout cannot hold: an unknown predicate or type, a constant argument, or more
    parameters, literals or predicate arguments than the layout allows.
    """

    def __init__(self, layout: OperatorLayout, name_backend: TextEmbeddingBackend | None = None):
        if (name_backend.dim if name_backend else 0) != layout.name_dim:
            raise ValueError(f"layout.name_dim is {layout.name_dim}; the name backend must match it")
        self.layout = layout
        self.name_backend = name_backend

    @property
    def dim(self) -> int:
        return self.layout.size

    def encode(self, operator: PDDLOperator) -> np.ndarray:
        layout = self.layout
        if len(operator.parameters) > layout.max_parameters:
            raise ValueError(f"{operator.name} has more than {layout.max_parameters} parameters")
        positions = {param.name: i for i, param in enumerate(operator.parameters)}
        literals = [
            self._literal(NEGATED_PRECONDITION if p.negated else PRECONDITION, p.predicate, positions)
            for p in operator.preconditions
        ] + [self._literal(DELETE if e.delete else ADD, e.predicate, positions) for e in operator.effects]
        if len(literals) > layout.max_literals:
            raise ValueError(f"{operator.name} has more than {layout.max_literals} literals")

        array = np.zeros(layout.size, dtype=np.float32)
        array[: len(operator.parameters)] = [self._id(layout.types, p.type, "type") for p in operator.parameters]
        literal_end = layout.max_parameters + layout.max_literals * layout.literal_width
        rows = array[layout.max_parameters : literal_end].reshape(layout.max_literals, layout.literal_width)
        for row, literal in zip(rows, sorted(literals)):
            row[: len(literal)] = literal
        if self.name_backend:
            array[literal_end:] = self.name_backend.embed_text(operator.name)
        return array

    def _literal(self, section: int, predicate: Predicate, positions: dict[str, int]) -> tuple[int, ...]:
        if len(predicate.args) > self.layout.max_predicate_arity:
            raise ValueError(f"{predicate} has more than {self.layout.max_predicate_arity} arguments")
        constants = [arg for arg in predicate.args if arg not in positions]
        if constants:
            raise ValueError(f"{predicate} has constant arguments {constants}; only parameters are supported")
        return (section, self._id(self.layout.predicates, predicate.name, "predicate"),
                *(positions[arg] + 1 for arg in predicate.args))

    @staticmethod
    def _id(vocabulary: tuple[str, ...], symbol: str, kind: str) -> int:
        if symbol not in vocabulary:
            raise ValueError(f"unknown {kind} {symbol!r}; known: {list(vocabulary)}")
        return vocabulary.index(symbol) + 1


class OperatorParts(NamedTuple):
    """A batch of structure arrays, unpacked (B = batch size)."""

    types: torch.Tensor  # (B, P) type id per parameter, 0 = no parameter
    sections: torch.Tensor  # (B, L) literal section, 0 = no literal
    predicates: torch.Tensor  # (B, L) predicate ids
    arguments: torch.Tensor  # (B, L, A) parameter position + 1 per argument, 0 = no argument
    name: torch.Tensor  # (B, D) name text embedding


def unpack(structure: torch.Tensor, layout: OperatorLayout) -> OperatorParts:
    """Policy side: (B, layout.size) structure arrays -> OperatorParts."""
    literal_end = layout.max_parameters + layout.max_literals * layout.literal_width
    ids = structure[:, :literal_end].round().long()
    rows = ids[:, layout.max_parameters :].reshape(-1, layout.max_literals, layout.literal_width)
    return OperatorParts(ids[:, : layout.max_parameters], rows[..., 0], rows[..., 1], rows[..., 2:], structure[:, literal_end:])


def mlp(in_dim: int, out_dim: int, hidden_dim: int | None = None) -> nn.Sequential:
    """The learned function used throughout: Linear -> ReLU -> Linear."""
    hidden_dim = hidden_dim or max(in_dim, out_dim)
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))


class ComponentEmbeddings(nn.Module):
    """Learnable embeddings of an operator's components, shared by all architectures (module docstring):
    type, variable (by position) and predicate embeddings; TYPED; LITERAL; and the name's learned map."""

    def __init__(self, layout: OperatorLayout, dim: int):
        super().__init__()
        self.layout = layout
        self.type_embedding = nn.Embedding(len(layout.types) + 1, dim, padding_idx=0)
        self.variable_embedding = nn.Embedding(max(layout.max_parameters, 1), dim)
        self.predicate_embedding = nn.Embedding(len(layout.predicates) + 1, dim, padding_idx=0)
        self.typed = mlp(2 * dim, dim)  # TYPED(variable, type)
        self.literal = mlp((1 + layout.max_predicate_arity) * dim, dim)  # LITERAL(predicate, arguments...)
        self.name = nn.Linear(layout.name_dim, dim) if layout.name_dim else None

    def typed_variables(self, parts: OperatorParts) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, P, dim) typed-variable embeddings (zero where there is no parameter) and (B, P) mask."""
        batch, count = parts.types.shape
        positions = torch.arange(count, device=parts.types.device).expand(batch, count)
        mask = parts.types > 0
        variables = self.typed(torch.cat([self.variable_embedding(positions), self.type_embedding(parts.types)], dim=-1))
        return variables * mask[..., None], mask

    def literals(self, parts: OperatorParts, variables: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, L, dim) literal embeddings (zero where there is no literal) and (B, L) mask."""
        batch, count, arity = parts.arguments.shape
        dim = variables.shape[-1]
        padded = torch.cat([torch.zeros_like(variables[:, :1]), variables], dim=1)  # index 0: no argument
        index = parts.arguments.reshape(batch, count * arity, 1).expand(-1, -1, dim)
        arguments = torch.gather(padded, 1, index).reshape(batch, count, arity * dim)
        mask = parts.sections > 0
        literals = self.literal(torch.cat([self.predicate_embedding(parts.predicates), arguments], dim=-1))
        return literals * mask[..., None], mask

    def tokens(self, parts: OperatorParts) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Every component as one token, in order [name], parameters, literals: (B, N, dim) tokens,
        (B, N) kinds (NAME_KIND, PARAMETER_KIND, or section + 1 for literals) and (B, N) mask."""
        variables, variable_mask = self.typed_variables(parts)
        literals, literal_mask = self.literals(parts, variables)
        tokens, kinds, masks = [variables, literals], [torch.full_like(parts.types, PARAMETER_KIND), parts.sections + 1], [variable_mask, literal_mask]
        if self.name is not None:
            batch = parts.types.shape[0]
            tokens.insert(0, self.name(parts.name)[:, None])
            kinds.insert(0, torch.full((batch, 1), NAME_KIND, dtype=torch.long, device=parts.types.device))
            masks.insert(0, torch.ones(batch, 1, dtype=torch.bool, device=parts.types.device))
        return torch.cat(tokens, dim=1), torch.cat(kinds, dim=1), torch.cat(masks, dim=1)


class TreeComposition(nn.Module):
    """Functional composition following the PDDL syntax (module docstring):
    OPERATOR(NAME, PARAMETERS, PRE(AND(preconditions)), EFF(AND(effects))), with NOT applied to negated
    preconditions and delete effects. Invariant to literal order and to unused (padded) slots."""

    def __init__(self, layout: OperatorLayout, component_dim: int, output_dim: int):
        super().__init__()
        dim = component_dim
        self.components = ComponentEmbeddings(layout, dim)
        self.negation = mlp(dim, dim)  # NOT
        self.conjunction_element, self.conjunction = mlp(dim, dim), mlp(dim, dim)  # AND: rho(sum(phi(x)))
        self.precondition = mlp(dim, dim)  # PRE
        self.effect = mlp(dim, dim)  # EFF
        self.parameter_element, self.parameter_set = mlp(dim, dim), mlp(dim, dim)  # PARAMETERS: rho(sum(phi(x)))
        self.operator = mlp((4 if layout.name_dim else 3) * dim, output_dim)  # OPERATOR

    def conjoin(self, literals: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """AND over the masked literals: (B, L, dim) -> (B, dim); an empty conjunction is AND({})."""
        return self.conjunction((self.conjunction_element(literals) * mask[..., None]).sum(dim=1))

    def forward(self, parts: OperatorParts) -> torch.Tensor:
        variables, variable_mask = self.components.typed_variables(parts)
        literals, literal_mask = self.components.literals(parts, variables)
        negated = (parts.sections == NEGATED_PRECONDITION) | (parts.sections == DELETE)
        literals = torch.where(negated[..., None], self.negation(literals), literals)
        precondition = self.precondition(self.conjoin(literals, literal_mask & (parts.sections <= NEGATED_PRECONDITION)))
        effect = self.effect(self.conjoin(literals, literal_mask & (parts.sections >= ADD)))
        parameters = self.parameter_set((self.parameter_element(variables) * variable_mask[..., None]).sum(dim=1))
        pieces = [parameters, precondition, effect]
        if self.components.name is not None:
            pieces.insert(0, self.components.name(parts.name))
        return self.operator(torch.cat(pieces, dim=-1))


class SlotAttention(nn.Module):
    """Slot attention (Locatello et al., 2020) with learned initial slots, so slots are deterministic and
    ordered. Each round, slots compete for the inputs (softmax over slots), take the attention-weighted
    mean of their inputs, and update through a GRU and a residual MLP. Masked inputs are ignored."""

    def __init__(self, dim: int, num_slots: int, iterations: int):
        super().__init__()
        self.initial_slots = nn.Parameter(torch.randn(num_slots, dim) * dim**-0.5)
        self.iterations, self.scale = iterations, dim**-0.5
        self.norm_inputs, self.norm_slots, self.norm_update = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.query, self.key, self.value = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.gru = nn.GRUCell(dim, dim)
        self.update = mlp(dim, dim, 2 * dim)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, N, dim) inputs and (B, N) mask -> (B, num_slots, dim) slots."""
        batch, _, dim = inputs.shape
        inputs = self.norm_inputs(inputs)
        keys, values = self.key(inputs), self.value(inputs)
        slots = self.initial_slots.expand(batch, -1, -1)
        for _ in range(self.iterations):
            logits = torch.einsum("bkd,bnd->bkn", self.query(self.norm_slots(slots)), keys) * self.scale
            attention = logits.softmax(dim=1) * mask[:, None, :]
            attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-8)
            updates = torch.einsum("bkn,bnd->bkd", attention, values)
            slots = self.gru(updates.reshape(-1, dim), slots.reshape(-1, dim)).reshape(batch, -1, dim)
            slots = slots + self.update(self.norm_update(slots))
        return slots


class SlotComposition(nn.Module):
    """Slot-attention composition (module docstring): component tokens plus kind embeddings, grouped into
    num_slots slots, flattened and projected to the operator embedding. Invariant to literal order and
    to unused (padded) slots."""

    def __init__(self, layout: OperatorLayout, component_dim: int, output_dim: int, num_slots: int = 4,
                 slot_iterations: int = 3):
        super().__init__()
        self.components = ComponentEmbeddings(layout, component_dim)
        self.kind = nn.Embedding(NUM_KINDS, component_dim)
        self.slot_attention = SlotAttention(component_dim, num_slots, slot_iterations)
        self.output = nn.Linear(num_slots * component_dim, output_dim)

    def forward(self, parts: OperatorParts) -> torch.Tensor:
        tokens, kinds, mask = self.components.tokens(parts)
        return self.output(self.slot_attention(tokens + self.kind(kinds), mask).flatten(1))


class CompositionalOperatorExtractor(BaseFeaturesExtractor):
    """Policy side: features = [grounding, observation, composition(operator structure)].

    The other observation parts pass through unchanged and the composed operator embedding (size
    output_dim) is appended last: the same layout as the other embedding conditions, so they differ
    only in how the operator embedding is produced. `architecture` is TreeComposition, SlotComposition
    or GeometricComposition, built with (layout, component_dim, output_dim, **architecture_kwargs).
    """

    def __init__(self, observation_space: gym.spaces.Dict, architecture: type[nn.Module], layout: OperatorLayout,
                 component_dim: int, output_dim: int, architecture_kwargs: dict | None = None):
        self.other_keys = sorted(key for key in observation_space.spaces if key != OPERATOR_EMBEDDING_KEY)
        other_dim = sum(int(observation_space[key].shape[0]) for key in self.other_keys)
        super().__init__(observation_space, features_dim=other_dim + output_dim)
        self.layout = layout
        self.composition = architecture(layout, component_dim, output_dim, **(architecture_kwargs or {}))

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [observations[key].float().flatten(start_dim=1) for key in self.other_keys]
        parts.append(self.composition(unpack(observations[OPERATOR_EMBEDDING_KEY].float(), self.layout)))
        return torch.cat(parts, dim=1)
