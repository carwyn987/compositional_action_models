"""Settled success needs the moved block at rest: lifting on and on through the raised band is not stopping in it."""

import numpy as np
import pytest

from cam.environments.fetch.fetch_scripted_policies import FetchScriptedPickupRaisedPolicy


def unstack_raised_environment():
    from main import parse_args
    from cam.experiments.environment_setup import setup_environment
    from cam.skills.registry import build_skill

    config = parse_args(["--skills", "unstack-raised", "--num-blocks", "2", "--reward", "shaped", "--seed", "0"])
    return setup_environment(config, [build_skill("unstack-raised", config)])


def successes(env, policy, episodes=3):
    """Settled successes in episodes of policy(obs, info) -> action (reset before each episode, if it has reset)."""
    wins = 0
    for episode in range(episodes):
        obs, info = env.reset(seed=50 + episode)
        getattr(policy, "reset", lambda: None)()
        done = truncated = False
        while not (done or truncated):
            obs, _, done, truncated, info = env.step(policy(obs, info))
        wins += int(info["is_success"])
    return wins


class ScriptedUnstackRaised:
    def __init__(self):
        self.policy = FetchScriptedPickupRaisedPolicy()

    def reset(self):
        self.policy.reset()

    def __call__(self, obs, info):
        return self.policy(obs, info, info["grounded_action_model"])


@pytest.mark.integration
def test_a_constant_lift_passes_through_the_raised_band_without_stopping_in_it():
    """Without its retreat, unstack's setup leaves the open gripper around the top block, so closing and lifting at a
    constant 1 cm per step carries it up through the raised band, held there for more than the 5 steps settled success asks. Without
    the at-rest condition (the control) that counts as unstack-raised; with it, it does not. The scripted policy,
    which stops in the band, still succeeds."""
    env = unstack_raised_environment()
    env.env.env.retreat = None

    def lift(obs, info):
        return np.array([0.0, 0.0, 0.2, -0.05])

    try:
        with_check, stopping = successes(env, lift), successes(env, ScriptedUnstackRaised())
        env.env.env.at_rest = None
        without_check = successes(env, lift)
    finally:
        env.close()
    assert without_check == 3 and with_check == 0 and stopping == 3
