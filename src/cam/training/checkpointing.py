"""Crash-safe saving: atomic file writes and periodic model checkpoints.

Each save writes a temporary file next to the target and then renames it over
the target (os.replace, atomic on POSIX). An interrupted save leaves the
previous file intact, never a partial one.
"""

import logging
import os
from pathlib import Path

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    os.replace(partial, path)


def atomic_save_model(model: BaseAlgorithm, path: Path) -> None:
    """Save an SB3 model to path (e.g. <run>/model.zip) without ever leaving a partial file there."""
    path = Path(path)
    partial = path.with_name(path.name + ".partial")  # has a suffix, so SB3 does not append ".zip"
    model.save(partial)
    os.replace(partial, path)


class PeriodicSaveCallback(BaseCallback):
    """Saves the model to model_path (atomically) every save_interval environment steps."""

    def __init__(self, save_interval: int, model_path: Path):
        super().__init__()
        self.save_interval = save_interval
        self.model_path = Path(model_path)
        self.next_save_step = save_interval

    def _on_training_start(self) -> None:
        self.next_save_step = self.num_timesteps + self.save_interval

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_save_step:
            atomic_save_model(self.model, self.model_path)
            logger.info("saved checkpoint at step %d to %s", self.num_timesteps, self.model_path)
            self.next_save_step += self.save_interval
        return True
