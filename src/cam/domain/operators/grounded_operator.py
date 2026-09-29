"""A PDDL operator with its parameters bound to objects, e.g. pickup(block0)."""

from dataclasses import dataclass
from functools import cached_property
from itertools import product

from cam.domain.operators.pddl import Effect, PDDLOperator, Precondition, Predicate


@dataclass(frozen=True)
class GroundedOperator:
    """An operator applied to specific objects.

    preconditions and effects are the operator's, with every variable replaced
    by its bound object: (holding ?o) becomes (holding block0). They are
    derived from operator and arguments on first access.

    `facts` passed to the checks are the ground predicates true in a state
    (closed world).
    """

    operator: PDDLOperator
    arguments: tuple[str, ...]  # objects bound to operator.parameters, in order

    def __post_init__(self):
        if len(self.arguments) != len(self.operator.parameters):
            raise ValueError(
                f"{self.operator.name} takes {len(self.operator.parameters)} arguments, "
                f"got {len(self.arguments)}: {self.arguments}"
            )

    @property
    def binding(self) -> dict[str, str]:
        return {param.name: obj for param, obj in zip(self.operator.parameters, self.arguments)}

    @cached_property
    def preconditions(self) -> tuple[Precondition, ...]:
        return tuple(Precondition(self._substitute(p.predicate), p.negated) for p in self.operator.preconditions)

    @cached_property
    def effects(self) -> tuple[Effect, ...]:
        return tuple(Effect(self._substitute(e.predicate), e.delete) for e in self.operator.effects)

    def preconditions_hold(self, facts: frozenset[Predicate]) -> bool:
        """True when this action can be applied: every positive precondition is
        in facts and every negated precondition is not."""
        return all((p.predicate in facts) != p.negated for p in self.preconditions)

    def effects_hold(self, facts: frozenset[Predicate]) -> bool:
        """True when this action's outcome is reached: every add effect is in
        facts and every delete effect is not."""
        return all((e.predicate in facts) != e.delete for e in self.effects)

    def _substitute(self, predicate: Predicate) -> Predicate:
        binding = self.binding
        return Predicate(predicate.name, tuple(binding.get(arg, arg) for arg in predicate.args))

    def __str__(self) -> str:
        return f"{self.operator.name}({', '.join(self.arguments)})"


def ground(operator: PDDLOperator, binding: dict[str, str]) -> GroundedOperator:
    """Bind every operator parameter to an object, e.g. {"?o": "block0"}."""
    parameter_names = [param.name for param in operator.parameters]
    missing = [name for name in parameter_names if name not in binding]
    extra = [name for name in binding if name not in parameter_names]
    if missing or extra:
        raise ValueError(
            f"binding for {operator.name} must bind exactly {parameter_names}; "
            f"missing {missing}, unknown {extra}"
        )
    return GroundedOperator(operator, tuple(binding[name] for name in parameter_names))


def groundings(operator: PDDLOperator, objects: dict[str, str]) -> list[GroundedOperator]:
    """Every assignment of distinct objects to the operator's parameters whose types match.

    objects maps object name to type, e.g. {"block0": "block", "block1": "block"}.
    """
    candidates = [[obj for obj, obj_type in objects.items() if obj_type == param.type] for param in operator.parameters]
    return [
        ground(operator, {param.name: obj for param, obj in zip(operator.parameters, assignment)})
        for assignment in product(*candidates)
        if len(set(assignment)) == len(assignment)
    ]


def applicable_groundings(
    operator: PDDLOperator, facts: frozenset[Predicate], objects: dict[str, str]
) -> list[GroundedOperator]:
    """The groundings whose preconditions hold in facts."""
    return [grounded for grounded in groundings(operator, objects) if grounded.preconditions_hold(facts)]
