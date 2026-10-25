"""unstack and unstack-raised start with the gripper moved away from the stacked block (FetchScriptedRetreatPolicy, a
random rise and sideways move), so the skill starts with an approach, as pickup does.

Control: without the retreat, their setup leaves the open gripper around the top block, and the constant action an
untrained policy outputs in every state (seed 0: close a little, rise 1.6 mm per step) succeeds at unstack on every
scene, before any training. With the retreat it succeeds on none, and the scripted unstack still succeeds on all.
"""

import numpy as np
import pytest

from cam.environments.skill_environment import SkillEnvironment

CONSTANT = np.array([0.02, 0.06, 0.05, -0.04])
SEEDS = range(2000, 2004)


def successes(policy: str, retreat: bool) -> list[bool]:
    from main import parse_args
    from cam.experiments.environment_setup import setup_environment
    from cam.skills.registry import build_skill
    from scripts.run_skill import SCRIPTED_POLICIES

    config = parse_args(["--skills", "unstack", "--num-blocks", "2", "--reward", "shaped", "--seed", "0"])
    env = setup_environment(config, [build_skill("unstack", config)])
    skill_environment = env
    while not isinstance(skill_environment, SkillEnvironment):
        skill_environment = skill_environment.env
    if not retreat:
        skill_environment.retreat = None
    results = []
    try:
        for seed in SEEDS:
            scripted = SCRIPTED_POLICIES["unstack"]()
            scripted.reset()
            obs, info = env.reset(seed=seed, options={"skill": "unstack"})
            done = truncated = False
            while not (done or truncated):
                action = scripted(obs, info, info["grounded_action_model"]) if policy == "scripted" else CONSTANT
                obs, _, done, truncated, info = env.step(np.asarray(action, dtype=np.float64))
            results.append(bool(info["is_success"]))
    finally:
        env.close()
    return results


@pytest.mark.integration
def test_unstack_starts_away_from_the_block_so_a_constant_close_and_lift_no_longer_succeeds():
    assert successes("constant", retreat=False) == [True] * len(SEEDS)
    assert successes("constant", retreat=True) == [False] * len(SEEDS)
    assert successes("scripted", retreat=True) == [True] * len(SEEDS)
