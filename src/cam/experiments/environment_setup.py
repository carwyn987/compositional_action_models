"""Builds the training / evaluation environment from a config dict (main.py's options).

Shared by main.py (training) and scripts/run_skill.py (running skills with
scripted, random or trained policies), so both see the same environment and
policy observation.
"""

import gymnasium as gym

from cam.environments.fetch import fetch_multiblock_environment
from cam.environments.fetch.fetch_env_state_annotation_wrapper import FetchEnvStateAnnotationWrapper
from cam.environments.fetch.fetch_predicate_evaluation_wrapper import (
    FETCH_PREDICATE_ARITIES,
    FetchPredicateEvaluationWrapper,
)
from cam.environments.fetch.fetch_rewards import FETCH_SHAPED_REWARDS
from cam.environments.fetch.fetch_scripted_policies import FetchScriptedPickupPolicy, FetchScriptedPutdownPolicy
from cam.environments.skill_environment import SkillEnvironment
from cam.representations.grounding_encoder import GroundingEncoder
from cam.representations.one_hot_operator_encoder import OneHotOperatorEncoder
from cam.representations.pddl_multi_hot_operator_encoder import PDDLMultiHotOperatorEncoder
from cam.representations.random_operator_encoder import RandomOperatorEncoder
from cam.skills.skill import Skill
from cam.training.policy_observation_wrapper import PolicyObservationWrapper

# Operator embedding methods: (config, skills) -> OperatorEncoder. multi-hot's vocabulary comes from the
# predicates the environment evaluates, so it is the same for any skill set (and for repaired operators).
OPERATOR_ENCODERS = {
    "multi-hot": lambda config, skills: PDDLMultiHotOperatorEncoder(
        FETCH_PREDICATE_ARITIES, config["max_operator_arity"]
    ),
    "one-hot": lambda config, skills: OneHotOperatorEncoder([skill.symbolic_action_model for skill in skills]),
    "random": lambda config, skills: RandomOperatorEncoder(config["operator_embedding_dim"], seed=config["seed"]),
}
SETUP_AND_EPISODE_STEP_LIMIT = 10_000  # inner limit; the episode limit is applied outside SkillEnvironment


def setup_environment(config: dict, skills: list[Skill]) -> gym.Env:
    """Fetch environment running episodes of the given skills, with observations for the policy
    (docs/policy_inputs.md). The wrappers are listed outermost first."""
    environment_kwargs = {}
    if config["num_blocks"] is not None:
        environment_kwargs["num_blocks"] = config["num_blocks"]
    if config["render"]:
        environment_kwargs.update(width=config["window_width"], height=config["window_height"])
    setup_policies = {"pickup": FetchScriptedPickupPolicy(), "putdown": FetchScriptedPutdownPolicy()}
    reward_functions = (
        {name: reward_class() for name, reward_class in FETCH_SHAPED_REWARDS.items()}
        if config["reward"] == "shaped"
        else {}
    )
    operator_encoder = OPERATOR_ENCODERS[config["operator_encoder"]](config, skills) # This is static, not trained, right now. TODO: Replace with trainable embeddings
    grounding_encoder = GroundingEncoder(
        object_types=FetchEnvStateAnnotationWrapper.OBJECT_TYPES,
        max_objects=fetch_multiblock_environment.MAX_BLOCKS,
        feature_dim=FetchEnvStateAnnotationWrapper.OBJECT_FEATURE_DIM,
        max_arity=config["max_operator_arity"],
    )

    env = PolicyObservationWrapper(  # policy observation: state, operator embedding, grounding
        gym.wrappers.TimeLimit(  # episode step limit (setup steps not counted)
            SkillEnvironment(  # per episode: skill, setup chain, grounding, success, reward
                FetchPredicateEvaluationWrapper(  # info["facts"]
                    FetchEnvStateAnnotationWrapper(  # info["environment_state", "objects", "object_features"]; drops goal keys
                        gym.make(  # the Fetch simulator
                            config["environment_id"],
                            render_mode="human" if config["render"] else None,
                            max_episode_steps=SETUP_AND_EPISODE_STEP_LIMIT,  # high, so setup is never cut off
                            **environment_kwargs,
                        )
                    )
                ),
                skills,
                setup_policies,
                reward_functions,
            ),
            config["max_steps_per_episode"],
        ),
        operator_encoder,
        grounding_encoder,
    )
    env.action_space.seed(config["seed"])
    return env
