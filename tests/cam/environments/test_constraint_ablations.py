"""--no-failures and --no-motion-penalties switch the constraints off, for an ablation of what they cost learning."""

import numpy as np
import pytest


def pickup_environment(*options):
    from main import parse_args
    from cam.experiments.environment_setup import setup_environment
    from cam.skills.registry import build_skill

    config = parse_args(["--skills", "pickup", "--num-blocks", "2", "--reward", "shaped", "--seed", "0", *options])
    return setup_environment(config, [build_skill("pickup", config)])


def press_into_the_table(env, steps=40):
    """Drive the gripper down into the table (which fails on excessive force within a few steps); the last info and
    the reward components seen."""
    env.reset(seed=3)
    seen = set()
    for _ in range(steps):
        _, _, done, truncated, info = env.step(np.array([0.0, 0.0, -1.0, 1.0]))
        seen |= set(info["reward_components"])
        if done or truncated:
            break
    return info, seen


@pytest.mark.integration
def test_without_failures_pressing_into_the_table_goes_on_and_without_motion_penalties_costs_no_force():
    """With the constraints (the control), pressing into the table ends the episode as a failure and costs force;
    without them, it goes on to the time limit, unfailed, and no motion penalty is in the reward."""
    for options, constrained in (((), True), (("--no-failures", "--no-motion-penalties"), False)):
        env = pickup_environment(*options)
        try:
            info, components = press_into_the_table(env)
        finally:
            env.close()
        assert info["failed"] is constrained
        assert ("force" in components) is constrained and ("smoothness" in components) is constrained
