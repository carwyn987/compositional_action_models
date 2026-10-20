"""scripts/wrong_embedding_control.py: the policy is given the other skill's operator, on this skill's episodes."""

import numpy as np
import pytest

from scripts.wrong_embedding_control import OperatorInput


@pytest.mark.integration
def test_the_policy_is_given_the_other_skills_operator_on_this_skills_episodes():
    """pickup-raised episodes carry pickup-raised's operator (the control: the two differ); wrapped, every
    observation carries pickup's instead, with a compositional encoder too (which identity aliases do not cover)."""
    from main import parse_args
    from cam.experiments.environment_setup import setup_environment
    from cam.skills.registry import build_skill

    config = parse_args(["--skills", "pickup", "pickup-raised", "--operator-encoder", "compositional",
                         "--compositional-name", "none", "--num-blocks", "2", "--seed", "0"])
    env = setup_environment(config, [build_skill(name, config) for name in ("pickup", "pickup-raised")])
    pickup = env.operator_encoder.encode(build_skill("pickup", config).symbolic_action_model)
    try:
        own, info = env.reset(seed=1, options={"skill": "pickup-raised"})
        swapped = OperatorInput(env, pickup)
        reset, _ = swapped.reset(seed=1, options={"skill": "pickup-raised"})
        stepped = swapped.step(np.zeros(4))[0]
    finally:
        env.close()
    assert info["skill"] == "pickup-raised" and not np.array_equal(own["operator_embedding"], pickup)
    assert np.array_equal(reset["operator_embedding"], pickup) and np.array_equal(stepped["operator_embedding"], pickup)
