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

    def intermediates(self, parts: OperatorParts) -> dict[str, torch.Tensor]:
        """Every stage of the composition by name; forward returns "operator". Per literal, (B, L, dim), zero for
        EMPTY rows: signed-literal (NOT applied to negated preconditions and delete effects) and conjunct (AND's
        element function phi of it). Per operator, (B, dim): and-sum pre / eff (each section's sum of conjuncts, so
        linear in its literals), and pre / eff (AND: rho of the sum; an empty conjunction is AND({})), pre, eff,
        parameters, and operator (B, output_dim)."""
        variables, variable_mask = self.components.typed_variables(parts)
        literals, literal_mask = self.components.literals(parts, variables)
        negated = (parts.kinds == NEGATED_PRECONDITION) | (parts.kinds == DELETE)
        literals = torch.where(negated[..., None], self.negation(literals), literals)
        stages, conjuncts = {"signed-literal": literals}, {}
        sections = {"pre": ((parts.kinds == PRECONDITION) | (parts.kinds == NEGATED_PRECONDITION), self.precondition),
                    "eff": ((parts.kinds == ADD) | (parts.kinds == DELETE), self.effect)}
        for section, (in_section, function) in sections.items():
            conjuncts[section] = self.conjunction_element(literals) * (literal_mask & in_section)[..., None]
            stages[f"and-sum {section}"] = conjuncts[section].sum(dim=1)
            stages[f"and {section}"] = self.conjunction(stages[f"and-sum {section}"])
            stages[section] = function(stages[f"and {section}"])  # PRE or EFF
        stages["conjunct"] = conjuncts["pre"] + conjuncts["eff"]  # each literal row is in one section, zero in the other
        stages["parameters"] = self.parameter_set((self.parameter_element(variables) * variable_mask[..., None]).sum(dim=1))
        pieces = [stages["parameters"], stages["pre"], stages["eff"]]
        if self.components.name is not None:
            pieces.insert(0, self.components.name(parts.name))
        stages["operator"] = self.operator(torch.cat(pieces, dim=-1))
        return stages

    def forward(self, parts: OperatorParts) -> torch.Tensor:
        return self.intermediates(parts)["operator"]
