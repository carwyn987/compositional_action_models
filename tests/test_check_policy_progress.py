"""scripts/check_policy_progress.py: a policy that gets to the goal slowly is told apart from one that is stuck.

Each case plays a scripted policy whose outcome is known: as written (success), slowed to a tenth of its speed (still
getting there: success after the training limit, or progressing), and frozen where training plateaus (stuck), so a
slow policy passing for stuck, or a stuck one for slow, fails a test."""

import numpy as np
import pytest

from cam.environments.fetch.fetch_scripted_policies import FetchScriptedPickupPolicy, FetchScriptedPushPolicy
from scripts.check_policy_progress import run_episode

MAX_STEPS = 300


def environment(skills):
    from main import parse_args
    from cam.experiments.environment_setup import setup_environment
    from cam.skills.registry import build_skill

    config = parse_args(["--skills", *skills, "--num-blocks", "2", "--reward", "shaped", "--seed", "0",
                         "--max-steps-per-episode", str(MAX_STEPS)])
    return setup_environment(config, [build_skill(name, config) for name in skills])


def scripted(policy, speed=1.0, frozen_in=()):
    """policy's actions with the motion scaled by speed, and no motion at all once its phase is in frozen_in."""
    policy.reset()

    def act(obs, info):
        action = np.asarray(policy(obs, info, info["grounded_action_model"]), dtype=np.float32)
        action[:3] *= 0.0 if policy.phase in frozen_in else speed
        return action

    return act


@pytest.mark.integration
def test_pickup_slow_is_not_stuck_and_grip_and_wait_is():
    """Gripping the block and never lifting it (the learnability runs' pickup plateau) is stuck, frozen, with the block
    gripped; the same motion at a tenth of the speed gets there after the training limit."""
    env = environment(["pickup"])
    try:
        normal = run_episode(env, "pickup", scripted(FetchScriptedPickupPolicy()), MAX_STEPS, seed=1)
        slow = run_episode(env, "pickup", scripted(FetchScriptedPickupPolicy(), speed=0.1), MAX_STEPS, seed=1)
        waiting = run_episode(env, "pickup", scripted(FetchScriptedPickupPolicy(), frozen_in=("lift",)), MAX_STEPS,
                              seed=1)
    finally:
        env.close()
    assert normal[0] == "success" and "after the training limit" not in normal[1]
    assert slow[0] == "success" and "after the training limit" in slow[1]
    assert waiting[0] == "stuck" and "frozen" in waiting[1] and "gripped" in waiting[1]


@pytest.mark.integration
def test_push_slow_is_progressing_and_waiting_behind_the_block_is_stuck():
    """A push at a tenth of the speed is still closing in when the episode ends: progressing, not stuck. Stopping
    behind the block without pushing is stuck."""
    env = environment(["push-left"])
    try:
        slow = run_episode(env, "push-left", scripted(FetchScriptedPushPolicy(), speed=0.1), MAX_STEPS, seed=1)
        waiting = run_episode(env, "push-left", scripted(FetchScriptedPushPolicy(), frozen_in=("push",)), MAX_STEPS,
                              seed=1)
    finally:
        env.close()
    assert slow[0] in ("progressing", "success")
    assert waiting[0] == "stuck" and "frozen" in waiting[1]
