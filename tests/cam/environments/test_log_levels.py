"""Per-step and per-episode chatter logs at DEBUG, so a run's log keeps what matters: failures and warnings."""

import logging

import numpy as np
import pytest

CHATTER = ("facts added", "facts at reset", "episode start", "setup start", "setup succeeded")


class Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records: list[tuple[int, str]] = []

    def emit(self, record):
        self.records.append((record.levelno, record.getMessage()))


@pytest.mark.integration
def test_fact_changes_and_setups_log_at_debug_and_failures_at_info():
    """An unstack episode (set up by pickup and stack) pressed into the table until it fails: the chatter is logged
    (the control: it is there at DEBUG), none of it at INFO, and the failure at INFO."""
    from main import parse_args
    from cam.experiments.environment_setup import setup_environment
    from cam.skills.registry import build_skill

    config = parse_args(["--skills", "unstack", "--num-blocks", "2", "--reward", "shaped", "--seed", "0"])
    env = setup_environment(config, [build_skill("unstack", config)])
    logger, handler = logging.getLogger("cam"), Records()
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        env.reset(seed=1)
        for _ in range(40):
            _, _, done, truncated, _ = env.step(np.array([0.0, 0.0, -1.0, 1.0]))
            if done or truncated:
                break
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
        env.close()
    chatter = [(levelno, message) for levelno, message in handler.records if message.startswith(CHATTER)]
    assert {message.split(":")[0] for _, message in chatter} >= {"facts added", "episode start", "setup succeeded"}
    assert all(levelno == logging.DEBUG for levelno, _ in chatter)
    assert any(levelno == logging.INFO and message.startswith("episode failed") for levelno, message in handler.records)
