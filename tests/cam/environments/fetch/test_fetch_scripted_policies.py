import pytest

from cam.environments.fetch.fetch_scripted_policies import (
    FetchScriptedPickupPolicy,
    FetchScriptedPickupRaisedPolicy,
    FetchScriptedPlaceBesidePolicy,
    FetchScriptedPushPolicy,
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
    "pickup-raised": FetchScriptedPickupRaisedPolicy,
    "unstack-raised": FetchScriptedPickupRaisedPolicy,
    "place-beside": FetchScriptedPlaceBesidePolicy,
    "push-left": FetchScriptedPushPolicy,
    "push-right": FetchScriptedPushPolicy,
    "push-front": FetchScriptedPushPolicy,
    "push-back": FetchScriptedPushPolicy,
    "push-left-front": FetchScriptedPushPolicy,
    "push-right-front": FetchScriptedPushPolicy,
    "push-left-back": FetchScriptedPushPolicy,
    "push-right-back": FetchScriptedPushPolicy,
    "push-beside": FetchScriptedPushPolicy,
}
CONFIG = {
    "symbolic_action_model_format": "pddl", "environment_id": "FetchMultiBlock-v0", "render": False,
    "reward": "shaped", "operator_encoder": "multi-hot", "operator_embedding_dim": 256, "max_operator_arity": 3,
    "max_steps_per_episode": 100, "seed": 0, "success_hold_steps": 5,
}


def run_scripted_episodes(skill_name: str, num_blocks: int, episodes: int) -> list:
    """(grounded action, success) of each episode, with the skill's scripted policy acting."""
    config = CONFIG | {"num_blocks": num_blocks, "skills": [skill_name]}
    if skill_name.startswith("push"):  # the scripted push is slow on purpose (gentle contact): allow it more steps
        config["max_steps_per_episode"] = 150
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
        results.append((info["grounded_action_model"], bool(info["is_success"]) or info.get("failure")))
    env.close()
    return results


@pytest.mark.integration
@pytest.mark.parametrize("skill_name", sorted(SKILL_REGISTRY))
@pytest.mark.parametrize("num_blocks", [2, 3])
def test_scripted_policy_succeeds_after_setup(skill_name, num_blocks):
    results = run_scripted_episodes(skill_name, num_blocks, episodes=3)
    # A scripted push goes in straight lines, so another block can lie on its path: knocking it aside is a
    # failure ("disturbed ..."), a limit of the scripted policy, not of the skill's setup or predicates.
    if skill_name.startswith("push"):
        results = [(grounded, outcome) for grounded, outcome in results if not str(outcome).startswith("disturbed")]
    allowed_failures = 1 if skill_name.startswith("push") and num_blocks > 2 else 0
    assert sum(outcome is not True for _, outcome in results) <= allowed_failures, results


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



def run_unstack_raised_lifting_both_blocks(frame: bool, check_failure: bool) -> dict:
    """unstack-raised by the scripted policy, with the bottom block carried along, tipped (3 cm off-centre, so the
    top block is not "on" it) once the top block is lifted: the exploit a trained policy found. The episode's
    last info, plus whether the operator's effects held at some step."""
    import numpy as np

    from cam.environments.skill_environment import SkillEnvironment

    config = CONFIG | {"num_blocks": 2, "skills": ["unstack-raised"], "success_hold_steps": 5}
    env = setup_environment(config, [build_skill("unstack-raised", config)])
    skill_environment = env
    while not isinstance(skill_environment, SkillEnvironment):
        skill_environment = skill_environment.env
    if not frame:
        skill_environment.frame_predicates = frozenset()
    if not check_failure:
        skill_environment.check_failure = None
    policy, sim = SCRIPTED_POLICIES["unstack-raised"](), env.unwrapped
    obs, info = env.reset(seed=0)
    policy.reset()
    top, bottom = info["grounded_action_model"].arguments
    effects_held = False
    for _ in range(config["max_steps_per_episode"]):
        obs, _, terminated, truncated, info = env.step(policy(obs, info, info["grounded_action_model"]))
        effects_held |= info["grounded_action_model"].effects_hold(info["facts"])
        if terminated or truncated:
            break
        top_position = info["environment_state"].block_positions[top]
        if top_position[2] > 0.49:  # lifted off: carry the bottom block along, tipped under the top one
            sim._set_block_position(int(bottom.removeprefix("block")), top_position - [0.03, 0.0, 0.05])
            sim._mujoco.mj_forward(sim.model, sim.data)
    env.close()
    return info | {"effects_held": effects_held}


@pytest.mark.integration
def test_unstack_raised_lifting_both_blocks_is_not_a_success():
    """End to end in the Fetch simulation. Without the frame and the failure check the exploit succeeds (so it
    is real); the frame alone rejects it (the bottom block left the table, which unstack-raised does not mention);
    with the default checks it fails as soon as the bottom block moves (disturbed)."""
    unprotected = run_unstack_raised_lifting_both_blocks(frame=False, check_failure=False)
    assert unprotected["effects_held"] and unprotected["is_success"]
    frame_only = run_unstack_raised_lifting_both_blocks(frame=True, check_failure=False)
    assert frame_only["effects_held"] and not frame_only["is_success"]
    default = run_unstack_raised_lifting_both_blocks(frame=True, check_failure=True)
    assert not default["is_success"] and default["failure"].startswith("disturbed")
