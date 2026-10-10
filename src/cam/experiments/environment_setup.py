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
from cam.environments.fetch.fetch_scripted_policies import (
    FetchScriptedPickupPolicy,
    FetchScriptedPlaceBesidePolicy,
    FetchScriptedPutdownPolicy,
    FetchScriptedStackPolicy,
)
from cam.environments.skill_environment import SkillEnvironment
from cam.representations.grounding_encoder import GroundingEncoder
from cam.representations.one_hot_operator_encoder import OneHotOperatorEncoder
from cam.representations.pddl_multi_hot_operator_encoder import PDDLMultiHotOperatorEncoder
from cam.representations.compositional.geometric import GeometricComposition
from cam.representations.compositional.slots import SlotComposition
from cam.representations.compositional.structure import OperatorLayout, StructuredOperatorEncoder
from cam.representations.compositional.tree import TreeComposition
from cam.representations.embedding_cache import DiskEmbeddingCache
from cam.representations.padded_operator_encoder import PaddedOperatorEncoder
from cam.representations.random_operator_encoder import RandomOperatorEncoder
from cam.representations.text_backends import TEXT_BACKENDS, CachedTextBackend
from cam.representations.text_operator_encoder import TextOperatorEncoder
from cam.skills.registry import SKILL_REGISTRY, build_skill
from cam.skills.skill import Skill
from cam.training.policy_observation_wrapper import PolicyObservationWrapper

# Operator embedding methods: (config, skills) -> OperatorEncoder. multi-hot's vocabulary comes from the
# predicates the environment evaluates, so it is the same for any skill set (and for repaired operators).
OPERATOR_ENCODERS = {
    # one-hot and multi-hot are zero-padded to --operator-embedding-dim, so every method has the same size
    "multi-hot": lambda config, skills: PaddedOperatorEncoder(
        PDDLMultiHotOperatorEncoder(FETCH_PREDICATE_ARITIES, config["max_operator_arity"]),
        config["operator_embedding_dim"],
    ),
    # one-hot indexes every registered skill's operator in registry order, so an operator keeps its index
    # whatever the training skills, and a skill added later has an index no training has used
    "one-hot": lambda config, skills: PaddedOperatorEncoder(
        OneHotOperatorEncoder(
            [build_skill(name, config).symbolic_action_model for name in SKILL_REGISTRY], identity_aliases(config)
        ),
        config["operator_embedding_dim"],
    ),
    "random": lambda config, skills: RandomOperatorEncoder(
        config["operator_embedding_dim"], seed=config["seed"], aliases=identity_aliases(config)
    ),
    "text": lambda config, skills: TextOperatorEncoder(text_backend(config), config["operator_text"]),
    # the operator's structure, composed into an embedding inside the policy (representations/compositional/)
    "compositional": lambda config, skills: StructuredOperatorEncoder(
        compositional_layout(config),
        text_backend(config, config["component_embedding_dim"]) if config["compositional_name"] == "text" else None,
    ),
}


def identity_aliases(config: dict) -> dict[str, str]:
    """--operator-identity-aliases NEW=OLD ... as {NEW: OLD} (configs saved before the option: none)."""
    return dict(alias.split("=", 1) for alias in config.get("operator_identity_aliases") or [])


# How CompositionalPolicyFeaturesExtractor composes the operator's components (--compositional-architecture).
COMPOSITIONAL_ARCHITECTURES = {"tree": TreeComposition, "slots": SlotComposition, "geometric": GeometricComposition}


def compositional_layout(config: dict) -> OperatorLayout:
    """The structure array layout: the predicates the environment evaluates and its object types, so it
    is the same for any skill set (and for new or repaired operators)."""
    return OperatorLayout.from_predicate_arities(
        FETCH_PREDICATE_ARITIES,
        FetchEnvStateAnnotationWrapper.OBJECT_TYPES,
        max_predicate_arity=config["max_predicate_arity"],
        max_parameters=config["max_operator_arity"],
        max_literals=config["max_operator_literals"],
        name_dim=config["component_embedding_dim"] if config["compositional_name"] == "text" else 0,
    )


def text_backend(config: dict, dim: int | None = None):
    """The configured text embedding backend, of size dim (default: the shared embedding size); backends
    other than the (deterministic, offline) mock are cached on disk so each text is embedded once."""
    backend = TEXT_BACKENDS[config["text_backend"]](dim or config["operator_embedding_dim"])
    if config["text_backend"] == "mock":
        return backend
    return CachedTextBackend(backend, DiskEmbeddingCache(config["text_embedding_cache"]))
SETUP_AND_EPISODE_STEP_LIMIT = 10_000  # inner limit; the episode limit is applied outside SkillEnvironment


def setup_environment(config: dict, skills: list[Skill]) -> gym.Env:
    """Fetch environment running episodes of the given skills, with observations for the policy
    (docs/policy_inputs.md). The wrappers are listed outermost first."""
    environment_kwargs = {}
    if config["num_blocks"] is not None:
        environment_kwargs["num_blocks"] = config["num_blocks"]
    if config["render"]:
        environment_kwargs.update(width=config["window_width"], height=config["window_height"])
    setup_policies = {
        "pickup": FetchScriptedPickupPolicy(),
        "putdown": FetchScriptedPutdownPolicy(),
        "stack": FetchScriptedStackPolicy(),
        "unstack": FetchScriptedPickupPolicy(),  # grasp and lift, from on top of the other block
        "pickup-raised": FetchScriptedPickupPolicy(),  # the pickup motion lifts the block 15 cm
        "unstack-raised": FetchScriptedPickupPolicy(),
        "place-beside": FetchScriptedPlaceBesidePolicy(),
    }
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
