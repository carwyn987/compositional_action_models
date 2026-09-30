"""Format-independent interfaces for symbolic action models, and grounding over them.

A SymbolicActionModel is lifted (e.g. pickup(?o)); ground() binds its
parameters to objects, giving a GroundedSymbolicActionModel (e.g.
pickup(block0)) that judges applicability and outcome from facts: the ground
predicates true in a state (closed world). Each format (PDDL, ...) implements
both classes; skills and training use only these interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import product
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cam.domain.operators.pddl import Predicate, TypedParameter


class GroundedSymbolicActionModel(ABC):
    arguments: tuple[str, ...]  # objects bound to the lifted model's parameters, in order

    @abstractmethod
    def preconditions_hold(self, facts: frozenset[Predicate]) -> bool:
        """True when this action can be applied in facts."""

    @abstractmethod
    def effects_hold(self, facts: frozenset[Predicate]) -> bool:
        """True when this action's outcome is reached in facts."""


class SymbolicActionModel(ABC):
    name: str
    parameters: tuple[TypedParameter, ...]  # typed slots, e.g. (?o - block)

    @abstractmethod
    def ground(self, binding: dict[str, str]) -> GroundedSymbolicActionModel:
        """Bind every parameter to an object, e.g. {"?o": "block0"}."""


def groundings(model: SymbolicActionModel, objects: dict[str, str]) -> list[GroundedSymbolicActionModel]:
    """Every assignment of distinct objects to the model's parameters whose types match.

    objects maps object name to type, e.g. {"block0": "block", "block1": "block"}.
    """
    candidates = [[obj for obj, obj_type in objects.items() if obj_type == param.type] for param in model.parameters]
    return [
        model.ground({param.name: obj for param, obj in zip(model.parameters, assignment)})
        for assignment in product(*candidates)
        if len(set(assignment)) == len(assignment)
    ]


def applicable_groundings(
    model: SymbolicActionModel, facts: frozenset[Predicate], objects: dict[str, str]
) -> list[GroundedSymbolicActionModel]:
    """The groundings whose preconditions hold in facts."""
    return [grounded for grounded in groundings(model, objects) if grounded.preconditions_hold(facts)]
