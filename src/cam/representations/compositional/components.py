"""Learnable component embeddings shared by every compositional architecture (see __init__.py)."""

import torch
from torch import nn

from cam.representations.compositional.structure import EMPTY, NAME, PARAMETER, OperatorLayout, OperatorParts


def mlp(in_dim: int, out_dim: int, hidden_dim: int | None = None) -> nn.Sequential:
    """The learned function used throughout: Linear -> ReLU -> Linear."""
    hidden_dim = hidden_dim or max(in_dim, out_dim)
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))


class ComponentEmbeddings(nn.Module):
    """Embedding store for the operator's components - contains the learned embeddings: type, variable
    (by position) and predicate embeddings, the learned functions TYPED and LITERAL that combine them,
    and the name's learned map. Every architecture builds on these. Components in EMPTY (padding)
    positions are exactly zero and masked."""

    def __init__(self, layout: OperatorLayout, dim: int):
        super().__init__()
        self.layout = layout
        self.type_embedding = nn.Embedding(len(layout.types) + 1, dim, padding_idx=0)
        self.variable_embedding = nn.Embedding(max(layout.max_parameters, 1), dim)
        self.predicate_embedding = nn.Embedding(len(layout.predicates) + 1, dim, padding_idx=0)
        # TODO: maybe make each type a functional embedding instead: a learned function per type applied to
        # the variable (type_function[type](variable)), replacing both type_embedding and TYPED.
        self.typed = mlp(2 * dim, dim)  # TYPED(variable, type)
        self.literal = mlp((1 + layout.max_predicate_arity) * dim, dim)  # LITERAL(predicate, arguments...)
        self.name = nn.Linear(layout.name_dim, dim) if layout.name_dim else None

    def typed_variables(self, parts: OperatorParts) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, P, dim) typed-variable embeddings and (B, P) mask of real parameters."""
        batch, count = parts.types.shape
        positions = torch.arange(count, device=parts.types.device).expand(batch, count)
        mask = parts.types > 0
        variables = self.typed(torch.cat([self.variable_embedding(positions), self.type_embedding(parts.types)], dim=-1))
        return variables * mask[..., None], mask

    def literals(self, parts: OperatorParts, variables: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, L, dim) literal embeddings and (B, L) mask of real literals. Each argument is the
        typed-variable embedding it refers to (zero for no argument), found with one gather."""
        batch, count, arity = parts.arguments.shape
        dim = variables.shape[-1]
        padded = torch.cat([torch.zeros_like(variables[:, :1]), variables], dim=1)  # index 0: no argument
        index = parts.arguments.reshape(batch, count * arity, 1).expand(-1, -1, dim)
        arguments = torch.gather(padded, 1, index).reshape(batch, count, arity * dim)
        mask = parts.kinds != EMPTY
        literals = self.literal(torch.cat([self.predicate_embedding(parts.predicates), arguments], dim=-1))
        return literals * mask[..., None], mask

    def tokens(self, parts: OperatorParts) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Every component as one token, in order [name], parameters, literals: (B, N, dim) tokens,
        (B, N) kinds (NAME, PARAMETER, or the literal's kind) and (B, N) mask."""
        variables, variable_mask = self.typed_variables(parts)
        literals, literal_mask = self.literals(parts, variables)
        tokens = [variables, literals]
        kinds = [torch.full_like(parts.types, PARAMETER), parts.kinds]
        masks = [variable_mask, literal_mask]
        if self.name is not None:
            batch, device = parts.types.shape[0], parts.types.device
            tokens.insert(0, self.name(parts.name)[:, None])
            kinds.insert(0, torch.full((batch, 1), NAME, dtype=torch.long, device=device))
            masks.insert(0, torch.ones(batch, 1, dtype=torch.bool, device=device))
        return torch.cat(tokens, dim=1), torch.cat(kinds, dim=1), torch.cat(masks, dim=1)
