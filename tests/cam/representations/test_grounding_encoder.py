import numpy as np
import pytest

from cam.domain.pddl.pddl_parser import parse_operator
from cam.representations.grounding_encoder import GroundingEncoder
from tests.cam.domain.conftest import LEGACY_OPERATORS

OBJECTS = {"block0": "block", "block1": "block", "block2": "block"}
FEATURES = {name: np.full(6, i + 1, dtype=np.float32) for i, name in enumerate(OBJECTS)}


@pytest.mark.unit
def test_slot_layout():
    """pickup(block2): [used | type one-hot | object index one-hot | features] (docs/policy_inputs.md example)."""
    pickup = parse_operator(LEGACY_OPERATORS["pickup"]).ground({"?o": "block2"})
    encoding = GroundingEncoder(["block"], max_objects=3, feature_dim=6, max_arity=1).encode(pickup, OBJECTS, FEATURES)
    np.testing.assert_array_equal(encoding, [1, 1, 0, 0, 1, 3, 3, 3, 3, 3, 3])


@pytest.mark.unit
def test_slots_follow_parameter_order_and_pad():
    """stack(block1, block0) fills slots 0 and 1 in parameter order; a third slot is zero padding."""
    stack = parse_operator(LEGACY_OPERATORS["stack"]).ground({"?o": "block1", "?b": "block0"})
    encoder = GroundingEncoder(["block"], max_objects=3, feature_dim=6, max_arity=3)
    slots = encoder.encode(stack, OBJECTS, FEATURES).reshape(3, encoder.slot_dim)
    assert slots[0, 3] == 1 and slots[0, -1] == 2  # block1
    assert slots[1, 2] == 1 and slots[1, -1] == 1  # block0
    assert not slots[2].any()


@pytest.mark.unit
def test_too_many_objects_is_an_error():
    """Scenes with more objects than max_objects cannot be encoded."""
    pickup = parse_operator(LEGACY_OPERATORS["pickup"]).ground({"?o": "block0"})
    with pytest.raises(ValueError):
        GroundingEncoder(["block"], max_objects=2, feature_dim=6, max_arity=1).encode(pickup, OBJECTS, FEATURES)
