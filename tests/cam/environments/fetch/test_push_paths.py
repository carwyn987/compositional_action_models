"""Pushes start with their path clear (push_path_clear, SkillEnvironment's feasible): with another block between the
pushed block and its goal, a push can only fail, by disturbing that block, so neither the scripted policy nor a
learned one could reach the success threshold.

Control: without the filter, the scripted diagonal push knocks the other block aside in 5 of the same 14 episodes.
"""

import numpy as np
import pytest

from cam.environments.skill_environment import SkillEnvironment

SKILL = "push-left-front"  # the diagonal: the longest path, blocked most often
SEEDS = range(300, 314)


def outcomes(filtered: bool) -> list[str]:
    from main import parse_args
    from cam.experiments.environment_setup import setup_environment
    from cam.skills.registry import build_skill
    from scripts.run_skill import SCRIPTED_POLICIES

    config = parse_args(["--skills", SKILL, "--num-blocks", "2", "--reward", "shaped", "--seed", "0"])
    env = setup_environment(config, [build_skill(SKILL, config)])
    skill_environment = env
    while not isinstance(skill_environment, SkillEnvironment):
        skill_environment = skill_environment.env
    if not filtered:
        skill_environment.feasible = None
    results = []
    try:
        for seed in SEEDS:
            policy = SCRIPTED_POLICIES[SKILL]()
            policy.reset()
            obs, info = env.reset(seed=seed, options={"skill": SKILL})
            done = truncated = False
            while not (done or truncated):
                action = np.asarray(policy(obs, info, info["grounded_action_model"]), dtype=np.float64)
                obs, _, done, truncated, info = env.step(action)
            results.append("success" if info["is_success"] else info["failure"] or "time-out")
    finally:
        env.close()
    return results


@pytest.mark.integration
def test_the_scripted_push_succeeds_on_every_scene_once_blocked_paths_are_ruled_out():
    assert outcomes(filtered=True) == ["success"] * len(SEEDS)
    assert any(outcome.startswith("disturbed") for outcome in outcomes(filtered=False))
