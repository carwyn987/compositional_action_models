#!/usr/bin/env python3
"""Run episodes of one skill in a SkillEnvironment and report setup, facts and success.

    ./scripts/run_skill.py                                 # putdown, set up by pickup, scripted policy
    ./scripts/run_skill.py --skill pickup --num-blocks 5 --render
    ./scripts/run_skill.py --policy random --episodes 10
"""

import argparse
import time

import gymnasium as gym

from cam.domain.action_model_library.loader import SYMBOLIC_ACTION_MODEL_FORMATS
from cam.environments.fetch import fetch_multiblock_environment
from cam.environments.fetch.fetch_env_state_annotation_wrapper import FetchEnvStateAnnotationWrapper
from cam.environments.fetch.fetch_predicate_evaluation_wrapper import FetchPredicateEvaluationWrapper
from cam.environments.fetch.fetch_scripted_policies import FetchScriptedPickupPolicy, FetchScriptedPutdownPolicy
from cam.environments.skill_environment import SkillEnvironment
from cam.logging_config import configure_logging
from cam.policies.policy import RandomPolicy
from cam.skills.registry import SKILL_REGISTRY, build_skill

SCRIPTED_POLICIES = {"pickup": FetchScriptedPickupPolicy, "putdown": FetchScriptedPutdownPolicy}
SETUP_AND_EPISODE_STEP_LIMIT = 10_000  # inner limit; the episode limit is applied outside SkillEnvironment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skill", choices=sorted(SKILL_REGISTRY), default="putdown")
    parser.add_argument("--policy", choices=["scripted", "random"], default="scripted")
    parser.add_argument("--symbolic-action-model-format", choices=sorted(SYMBOLIC_ACTION_MODEL_FORMATS), default="pddl")
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=100, help="episode step limit (setup steps excluded)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.02, help="seconds between rendered frames")
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=960)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    skill = build_skill(args.skill, {"symbolic_action_model_format": args.symbolic_action_model_format})
    base_env = gym.make(
        fetch_multiblock_environment.ENVIRONMENT_ID,
        num_blocks=args.num_blocks,
        render_mode="human" if args.render else None,
        width=args.window_width,
        height=args.window_height,
        max_episode_steps=SETUP_AND_EPISODE_STEP_LIMIT,
    )
    setup_policies = {name: policy_class() for name, policy_class in SCRIPTED_POLICIES.items()}
    skill_env = SkillEnvironment(
        FetchPredicateEvaluationWrapper(FetchEnvStateAnnotationWrapper(base_env)), [skill], setup_policies
    )
    env = gym.wrappers.TimeLimit(skill_env, args.max_steps)
    env.action_space.seed(args.seed)
    policy = SCRIPTED_POLICIES[skill.name]() if args.policy == "scripted" else RandomPolicy(env.action_space)

    successes = 0
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed if episode == 0 else None)
            setup = ", ".join(f"{grounded} ({steps} steps)" for grounded, steps in info["setup"]) or "none"
            print(f"=== episode {episode} target={info['grounded_action_model']} setup: {setup}")

            policy.reset()
            for t in range(args.max_steps):
                obs, reward, terminated, truncated, info = env.step(policy(obs, info, info["grounded_action_model"]))
                if args.render:
                    time.sleep(args.sleep)
                if terminated or truncated:
                    break
            successes += int(info["is_success"])
            print(f"    success={bool(info['is_success'])} steps={t + 1} reward={reward}")
    finally:
        env.close()
    print(f"success rate: {successes}/{args.episodes}")


if __name__ == "__main__":
    main()
