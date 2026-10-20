#!/usr/bin/env python3
"""Wrong-embedding control: does a trained policy use the operator to tell two skills apart?

    ./scripts/wrong_embedding_control.py outputs/<run> --skill unstack-raised --as unstack
    ./scripts/wrong_embedding_control.py outputs/<run> --skill pickup-raised --as pickup --episodes 50

Plays the run's policy (deterministic) on --skill's episodes twice, on the run's evaluation scenes: given --skill's own
operator input, then --as's. Skills that start from the same scenes with the same grounding (unstack and
unstack-raised, pickup and pickup-raised) differ only in their operator, so if success and behaviour (how high the
moved block goes, how long the episodes last) hardly change, the policy does not use the operator to tell them apart:
a success on a held-out operator then comes from behaviour it shares with the other, not from reading the operator.
Works with every operator encoder, including compositional ones (which --operator-identity-aliases does not cover).
"""

import argparse
import json
import logging
from pathlib import Path

import gymnasium as gym
import numpy as np

from cam.environments.fetch.fetch_predicate_evaluation_wrapper import TABLE_REST_Z
from cam.experiments.environment_setup import setup_environment
from cam.policies.stable_baselines3_policy import ALGORITHMS
from cam.skills.registry import build_skill

EVALUATION_SEED_OFFSET = 10_000  # main.py evaluates on scenes seeded with the run's seed + this


class OperatorInput(gym.ObservationWrapper):
    """Replaces obs["operator_embedding"] with a fixed operator input (None: leaves it as it is)."""

    def __init__(self, env: gym.Env, operator_input: np.ndarray | None):
        super().__init__(env)
        self.operator_input = operator_input

    def observation(self, obs):
        return obs if self.operator_input is None else {**obs, "operator_embedding": self.operator_input}


def play(run: Path, skill: str, given: str, episodes: int) -> str:
    """One line: the run's policy on skill's episodes, given the operator input of skill given."""
    config = json.loads((run / "config.json").read_text()) | {"render": False}
    model = ALGORITHMS[config["algorithm"]].load(run / "model.zip", device="cpu")
    env = setup_environment(config, [build_skill(name, config) for name in dict.fromkeys([*config["skills"], skill, given])])
    operator_input = None if given == skill else env.operator_encoder.encode(build_skill(given, config).symbolic_action_model)
    env = OperatorInput(env, operator_input)
    wins, heights, lengths = 0, [], []
    try:
        for episode in range(episodes):
            seed = config["seed"] + EVALUATION_SEED_OFFSET if episode == 0 else None
            obs, info = env.reset(seed=seed, options={"skill": skill})
            block, top = info["grounded_action_model"].arguments[0], 0.0
            for step in range(1, config["max_steps_per_episode"] + 1):
                obs, _, terminated, truncated, info = env.step(model.predict(obs, deterministic=True)[0])
                top = max(top, float(info["environment_state"].block_positions[block][2]) - TABLE_REST_Z)
                if terminated or truncated:
                    break
            wins += int(info["is_success"])
            heights.append(top)
            lengths.append(step)
    finally:
        env.close()
    return (f"{skill} episodes given {given}'s operator: success {wins}/{episodes}, highest lift "
            f"{100 * np.mean(heights):.0f} ± {100 * np.std(heights):.0f} cm above the table, length {np.mean(lengths):.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="run directory (with config.json and model.zip)")
    parser.add_argument("--skill", required=True, help="the skill whose episodes are played")
    parser.add_argument("--as", dest="as_skill", required=True, help="the skill whose operator input is given instead")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    for given in (args.skill, args.as_skill):
        print(play(args.run, args.skill, given, args.episodes), flush=True)


if __name__ == "__main__":
    main()
