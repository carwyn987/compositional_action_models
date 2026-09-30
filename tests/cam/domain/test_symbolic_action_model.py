from dataclasses import dataclass

import pytest

from cam.domain.symbols import Predicate, TypedParameter
from cam.domain.symbolic_action_model import (
    GroundedSymbolicActionModel,
    SymbolicActionModel,
    applicable_groundings,
)


@dataclass(frozen=True)
class GroundedLookupModel(GroundedSymbolicActionModel):
    """A non-PDDL format: applicable when (ready ?x) holds, done when (done ?x) holds."""

    arguments: tuple[str, ...]

    def preconditions_hold(self, facts):
        return Predicate("ready", self.arguments) in facts

    def effects_hold(self, facts):
        return Predicate("done", self.arguments) in facts


@dataclass(frozen=True)
class LookupModel(SymbolicActionModel):
    name: str = "process"
    parameters: tuple = (TypedParameter("?x", "item"),)

    def ground(self, binding):
        return GroundedLookupModel((binding["?x"],))


@pytest.mark.unit
def test_applicable_groundings_work_for_any_format():
    """Grounding and applicability use only the interfaces, so a non-PDDL model works unchanged."""
    facts = frozenset({Predicate("ready", ("item1",))})
    grounded = applicable_groundings(LookupModel(), facts, {"item0": "item", "item1": "item", "block0": "block"})
    assert [g.arguments for g in grounded] == [("item1",)]

