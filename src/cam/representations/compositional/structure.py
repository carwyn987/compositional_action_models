"""Serializing an operator into the observation: the structure array and its layout.

Every component of an operator has a *kind*, one global index used everywhere (the structure array's
literal rows, the slots architecture's kind embeddings, the geometric architecture's grid columns).
EMPTY marks padding: the structure array has a fixed size (room for max_parameters parameters and
max_literals literals), so the room an operator does not use is filled with EMPTY = 0 and masked out
by every architecture. Type and predicate ids likewise use 0 for "no type / no predicate".
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from cam.domain.pddl.pddl import PDDLOperator
from cam.domain.symbols import Predicate
from cam.representations.operator_encoder import OperatorEncoder
from cam.representations.text_backends import TextEmbeddingBackend

EMPTY, NAME, PARAMETER, PRECONDITION, NEGATED_PRECONDITION, ADD, DELETE = range(7)
NUM_KINDS = 7


@dataclass(frozen=True)
class OperatorLayout:
    """Sizes and vocabularies of the structure array, shared by StructuredOperatorEncoder and the
    architectures so both agree. Predicate and type ids are their vocabulary index + 1 (0 = empty).

    The maxima are fixed settings (like the other embeddings' maxima), not derived from the current
    operators, so new or modified operators fit the same layout; operators exceeding them are rejected.
    """

    predicates: tuple[str, ...]
    types: tuple[str, ...]
    max_predicate_arity: int
    max_parameters: int
    max_literals: int
    name_dim: int = 0

    @classmethod
    def from_predicate_arities(
        cls, predicate_arities: dict[str, int], types: list[str], max_predicate_arity: int, max_parameters: int,
        max_literals: int, name_dim: int = 0,
    ) -> "OperatorLayout":
        """Layout over the given predicates (e.g. those the environment evaluates); raises ValueError if
        a predicate has more arguments than max_predicate_arity."""
        too_long = {name: arity for name, arity in predicate_arities.items() if arity > max_predicate_arity}
        if too_long:
            raise ValueError(f"predicates {too_long} have more than max_predicate_arity={max_predicate_arity} arguments")
        return cls(tuple(sorted(predicate_arities)), tuple(types), max_predicate_arity, max_parameters, max_literals, name_dim)

    @property
    def literal_width(self) -> int:
        return 2 + self.max_predicate_arity

    @property
    def size(self) -> int:
        return self.max_parameters + self.max_literals * self.literal_width + self.name_dim


class StructuredOperatorEncoder(OperatorEncoder):
    """An operator -> its structure array, decoded by CompositionalOperatorExtractor in the policy.

    Structure array (ids stored as floats; 0 = EMPTY / none):
        [type id per parameter (P)] [per literal: kind, predicate id, argument refs (L x (2 + A))]
        [name text embedding (D)]
    with P = max_parameters, L = max_literals, A = max_predicate_arity, D = name_dim.

    How it is built:
        1. Variables are replaced by their parameter position + 1 (?o -> 1), so variable names never
           matter and argument refs line up with grounding slots.
        2. Each literal becomes (kind, predicate id, argument refs...), e.g. (on-table ?o) as a
           precondition -> (PRECONDITION, id of on-table, 1); a literal with fewer than A arguments
           leaves the remaining refs 0.
        3. Parameter type ids fill the first P entries, the literals fill the rows *sorted*, so the same
           operator written in any literal order gives the same array, and the name's text embedding
           (if name_dim > 0) fills the end. Unused parameter entries and literal rows stay 0 (EMPTY).
    Example (pickup, legacy PDDL, max_literals = 8): types [1, 0, 0]; rows
        [3 1 1 0] pre (clear ?o)        [3 2 0 0] pre (gripper-empty)     [3 5 1 0] pre (on-table ?o)
        [5 3 1 0] add (holding ?o)      [6 2 0 0] del (gripper-empty)     [6 5 1 0] del (on-table ?o)
        [0 0 0 0] [0 0 0 0] empty
    (predicate ids: clear 1, gripper-empty 2, holding 3, on 4, on-table 5).

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

    def _literal(self, kind: int, predicate: Predicate, positions: dict[str, int]) -> tuple[int, ...]:
        if len(predicate.args) > self.layout.max_predicate_arity:
            raise ValueError(f"{predicate} has more than {self.layout.max_predicate_arity} arguments")
        constants = [arg for arg in predicate.args if arg not in positions]
        if constants:
            raise ValueError(f"{predicate} has constant arguments {constants}; only parameters are supported")
        return (kind, self._id(self.layout.predicates, predicate.name, "predicate"),
                *(positions[arg] + 1 for arg in predicate.args))

    @staticmethod
    def _id(vocabulary: tuple[str, ...], symbol: str, description: str) -> int:
        if symbol not in vocabulary:
            raise ValueError(f"unknown {description} {symbol!r}; known: {list(vocabulary)}")
        return vocabulary.index(symbol) + 1


class OperatorParts(NamedTuple):
    """A batch of structure arrays, unpacked (B = batch size)."""

    types: torch.Tensor  # (B, P) type id per parameter, 0 = no parameter
    kinds: torch.Tensor  # (B, L) literal kind (PRECONDITION ... DELETE), EMPTY = no literal
    predicates: torch.Tensor  # (B, L) predicate ids
    arguments: torch.Tensor  # (B, L, A) parameter position + 1 per argument, 0 = no argument
    name: torch.Tensor  # (B, D) name text embedding


def unpack(structure: torch.Tensor, layout: OperatorLayout) -> OperatorParts:
    """(B, layout.size) structure arrays -> OperatorParts (ids rounded back to integers)."""
    literal_end = layout.max_parameters + layout.max_literals * layout.literal_width
    ids = structure[:, :literal_end].round().long()
    rows = ids[:, layout.max_parameters :].reshape(-1, layout.max_literals, layout.literal_width)
    return OperatorParts(ids[:, : layout.max_parameters], rows[..., 0], rows[..., 1], rows[..., 2:], structure[:, literal_end:])
