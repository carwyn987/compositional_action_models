#!/usr/bin/env python3
"""Where a training step's time goes, for one training configuration: the simulator step, the setup steps every
reset runs, policy inference, the SAC gradient update, and evaluation. Off-policy training alternates one step of
collection with one gradient update, so whichever is larger bounds the steps per second.

    .venv/bin/python scripts/profile_training.py --device cuda                      # pickup, unstack, pickup-raised
    .venv/bin/python scripts/profile_training.py --device cpu --operator-encoder compositional --compositional-name none
    .venv/bin/python scripts/profile_training.py --device cuda --skills stack        # any main.py options

Runs alongside other training share the CPU and GPU with it, so measure on an idle machine for one run's own
numbers, or during an experiment for what each of its runs gets.
"""

import cProfile
import io
import logging
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import parse_args, policy_kwargs  # noqa: E402
from cam.experiments.environment_setup import setup_environment  # noqa: E402
from cam.skills.registry import build_skill  # noqa: E402
from cam.training.stable_baselines3_trainer import ALGORITHMS, HYPERPARAMETERS  # noqa: E402

DEFAULTS = ["--skills", "pickup", "unstack", "pickup-raised", "--algorithm", "mtsac", "--reward", "shaped",
            "--num-blocks", "2", "--seed", "0"]
EPISODE_LENGTHS = (30, 60, 100)  # trained policies finish in about 25-60 steps; time-outs take 100


def main() -> None:
    logging.disable(logging.CRITICAL)
    config = parse_args(DEFAULTS + sys.argv[1:])
    env = setup_environment(config, [build_skill(name, config) for name in config["skills"]])
    rng = np.random.default_rng(0)

    # Resets (with their setup chains) and steps, with random actions.
    reset_times, setup_steps, step_times = {}, {}, []
    for episode in range(30):
        skill = config["skills"][episode % len(config["skills"])]
        start = time.perf_counter()
        obs, info = env.reset(seed=episode, options={"skill": skill})
        reset_times.setdefault(skill, []).append(time.perf_counter() - start)
        setup_steps.setdefault(skill, []).append(sum(steps for _, steps in info["setup"]))
        for _ in range(40):
            start = time.perf_counter()
            obs, _, done, truncated, info = env.step(rng.uniform(-1, 1, 4))
            step_times.append(time.perf_counter() - start)
            if done or truncated:
                break
    step = float(np.median(step_times))
    reset = float(np.mean([np.median(times) for times in reset_times.values()]))

    # What a step spends its time on.
    profile = cProfile.Profile()
    obs, info = env.reset(seed=99, options={"skill": config["skills"][0]})
    profile.enable()
    for _ in range(300):
        obs, _, done, truncated, info = env.step(rng.uniform(-1, 1, 4))
        if done or truncated:
            obs, info = env.reset(options={"skill": config["skills"][0]})
    profile.disable()
    profile_text = io.StringIO()
    pstats.Stats(profile, stream=profile_text).sort_stats("tottime").print_stats(6)

    # The model as main.py builds it: inference per step, and gradient updates on a filled replay buffer.
    model = ALGORITHMS[config["algorithm"]](
        "MultiInputPolicy", env, seed=0, verbose=0, device=config["device"], policy_kwargs=policy_kwargs(config) or {},
        **(HYPERPARAMETERS[config["algorithm"]] | {"learning_starts": 1500}))
    model.learn(total_timesteps=1500)  # collects only: no updates before learning_starts
    inference = []
    for _ in range(200):
        start = time.perf_counter()
        model.predict(obs, deterministic=False)
        inference.append(time.perf_counter() - start)
    predict = float(np.median(inference))

    def synchronize() -> None:
        if model.device.type == "cuda":
            torch.cuda.synchronize()

    model.train(gradient_steps=20, batch_size=256)  # warm-up
    synchronize()
    start = time.perf_counter()
    model.train(gradient_steps=200, batch_size=256)
    synchronize()
    update = (time.perf_counter() - start) / 200

    print(f"device {model.device}, torch threads {torch.get_num_threads()}, skills {' '.join(config['skills'])}, "
          f"operator encoder {config['operator_encoder']}")
    print(f"  simulator step, all wrappers:   {1000 * step:6.2f} ms")
    for skill in config["skills"]:
        print(f"  reset of {skill:20s} {1000 * np.median(reset_times[skill]):7.1f} ms ({np.mean(setup_steps[skill]):.0f} setup steps)")
    print(f"  policy inference:               {1000 * predict:6.2f} ms")
    print(f"  SAC gradient update (batch 256): {1000 * update:6.2f} ms")
    for length in EPISODE_LENGTHS:
        collect = step + predict + reset / length
        print(f"  episodes of {length:3d} steps: collection {1000 * collect:5.2f} ms per step, update {1000 * update:5.2f} ms"
              f" -> {1 / (collect + update):4.0f} steps/s; collection is {collect / (collect + update):.0%} of the time")
    evaluation = 2 * config["eval_episodes"] * len(config["skills"]) * (reset + 60 * (step + predict))
    print(f"  one evaluation ({config['eval_episodes']} episodes x deterministic and stochastic x "
          f"{len(config['skills'])} skills, 60 steps each): {evaluation:.0f} s")
    print("\nThe simulator step's own time, by function:")
    print("\n".join(line[:150] for line in profile_text.getvalue().splitlines()[6:14]))


if __name__ == "__main__":
    main()
