"""Tree architecture: learned functions composing the components along the PDDL syntax (see __init__.py)."""

import torch
from torch import nn

from cam.representations.compositional.components import ComponentEmbeddings, mlp
from cam.representations.compositional.structure import (
    ADD,
    DELETE,
    NEGATED_PRECONDITION,
    PRECONDITION,
    OperatorLayout,
    OperatorParts,
)


class TreeComposition(nn.Module):
    """OPERATOR(NAME, PARAMETERS, PRE(AND(preconditions)), EFF(AND(effects))), with NOT applied to negated
    preconditions and delete effects; AND and PARAMETERS are DeepSets (rho(sum(phi(x)))). Invariant to
    literal order and to EMPTY (padding) positions."""

    def __init__(self, layout: OperatorLayout, component_dim: int, output_dim: int):
        super().__init__()
        dim = component_dim
        self.components = ComponentEmbeddings(layout, dim)
        self.negation = mlp(dim, dim)  # NOT
        self.conjunction_element, self.conjunction = mlp(dim, dim), mlp(dim, dim)  # AND
        self.precondition = mlp(dim, dim)  # PRE
        self.effect = mlp(dim, dim)  # EFF
        self.parameter_element, self.parameter_set = mlp(dim, dim), mlp(dim, dim)  # PARAMETERS
        self.operator = mlp((4 if layout.name_dim else 3) * dim, output_dim)  # OPERATOR

    def conjoin(self, literals: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """AND over the masked literals: (B, L, dim) -> (B, dim); an empty conjunction is AND({})."""
        return self.conjunction((self.conjunction_element(literals) * mask[..., None]).sum(dim=1))

    def forward(self, parts: OperatorParts) -> torch.Tensor:
        variables, variable_mask = self.components.typed_variables(parts)
        literals, literal_mask = self.components.literals(parts, variables)
        negated = (parts.kinds == NEGATED_PRECONDITION) | (parts.kinds == DELETE)
        literals = torch.where(negated[..., None], self.negation(literals), literals)
        is_precondition = (parts.kinds == PRECONDITION) | (parts.kinds == NEGATED_PRECONDITION)
        is_effect = (parts.kinds == ADD) | (parts.kinds == DELETE)
        precondition = self.precondition(self.conjoin(literals, literal_mask & is_precondition))
        effect = self.effect(self.conjoin(literals, literal_mask & is_effect))
        parameters = self.parameter_set((self.parameter_element(variables) * variable_mask[..., None]).sum(dim=1))
        pieces = [parameters, precondition, effect]
        if self.components.name is not None:
            pieces.insert(0, self.components.name(parts.name))
        return self.operator(torch.cat(pieces, dim=-1))
