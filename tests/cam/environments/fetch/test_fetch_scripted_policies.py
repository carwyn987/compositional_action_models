import pytest

from cam.environments.fetch.fetch_scripted_policies import (
    FetchScriptedPickupPolicy,
    FetchScriptedPutdownPolicy,
    FetchScriptedStackPolicy,
)
from cam.experiments.environment_setup import setup_environment
from cam.skills.registry import SKILL_REGISTRY, build_skill

SCRIPTED_POLICIES = {
    "pickup": FetchScriptedPickupPolicy,
    "putdown": FetchScriptedPutdownPolicy,
    "stack": FetchScriptedStackPolicy,
    "unstack": FetchScriptedPickupPolicy,
}
CONFIG = {
    "symbolic_action_model_format": "pddl", "environment_id": "FetchMultiBlock-v0", "render": False,
    "reward": "shaped", "operator_encoder": "multi-hot", "operator_embedding_dim": 128, "max_operator_arity": 3,
    "max_steps_per_episode": 100, "seed": 0,
}


def run_scripted_episodes(skill_name: str, num_blocks: int, episodes: int) -> list:
    """(grounded action, success) of each episode, with the skill's scripted policy acting."""
    config = CONFIG | {"num_blocks": num_blocks, "skills": [skill_name]}
    env = setup_environment(config, [build_skill(skill_name, config)])
    policy = SCRIPTED_POLICIES[skill_name]()
    results = []
    for episode in range(episodes):
        obs, info = env.reset(seed=episode)
        policy.reset()
        for _ in range(config["max_steps_per_episode"]):
            obs, _, terminated, truncated, info = env.step(policy(obs, info, info["grounded_action_model"]))
            if terminated or truncated:
                break
        results.append((info["grounded_action_model"], bool(info["is_success"])))
    env.close()
    return results


@pytest.mark.integration
@pytest.mark.parametrize("skill_name", sorted(SKILL_REGISTRY))
@pytest.mark.parametrize("num_blocks", [2, 3])
def test_scripted_policy_succeeds_after_setup(skill_name, num_blocks):
    results = run_scripted_episodes(skill_name, num_blocks, episodes=3)
    assert all(success for _, success in results), results


@pytest.mark.integration
def test_stack_targets_vary_between_episodes():
    """Setup picks up a random block and the grounding picks a random base, so the pair varies."""
    groundings = {grounded.arguments for grounded, _ in run_scripted_episodes("stack", num_blocks=3, episodes=8)}
    assert len(groundings) > 1


@pytest.mark.integration
@pytest.mark.parametrize("skill_name", ["stack", "unstack"])
def test_two_block_skills_reject_a_one_block_scene(skill_name):
    with pytest.raises(ValueError, match="needs distinct objects"):
        run_scripted_episodes(skill_name, num_blocks=1, episodes=1)
