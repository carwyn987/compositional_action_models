"""New PPO models start with action noise low enough to hold a grasped block."""

import numpy as np
import pytest
import torch

from cam.training.stable_baselines3_trainer import PPO_LOG_STD_INIT, train_stable_baselines3


@pytest.mark.integration
def test_a_new_ppo_model_starts_with_a_standard_deviation_of_about_0_37(tmp_path):
    """Stable-Baselines3's default would be 1 (log 0), at which a noisy gripper command lets go of a grasped block."""
    from main import parse_args
    from cam.experiments.environment_setup import setup_environment
    from cam.skills.registry import build_skill

    config = parse_args(["--skills", "pickup", "--algorithm", "ppo", "--num-blocks", "2", "--seed", "0"])
    env = setup_environment(config, [build_skill("pickup", config)])
    try:
        model = train_stable_baselines3(env, "ppo", 1, tmp_path, hyperparameter_overrides={"n_steps": 64,
                                                                                          "batch_size": 64,
                                                                                          "learning_rate": 0.0})
    finally:
        env.close()
    std = torch.exp(model.policy.log_std).detach().numpy()
    assert np.allclose(std, np.exp(PPO_LOG_STD_INIT)) and np.exp(PPO_LOG_STD_INIT) < 0.4
