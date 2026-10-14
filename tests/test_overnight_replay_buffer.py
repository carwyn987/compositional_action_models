"""scripts/overnight_runs.py: repairs share their rq1 run's replay buffer through a hard link."""

import os
import shutil

import pytest

from scripts.overnight_runs import link_replay_buffer
from cam.training.checkpointing import atomic_write_text


@pytest.mark.unit
def test_repairs_link_the_rq1_buffer_and_saving_never_writes_through_the_link(tmp_path):
    """The repair's copy links the replay buffer (no second 1 GB file) and copies the rest. Writing the linked file in
    place would change the rq1 run's buffer too (the control); saving writes a new file and renames it, which
    replaces the link and leaves the rq1 buffer as it was."""
    rq1 = tmp_path / "rq1"
    rq1.mkdir()
    (rq1 / "replay_buffer.pkl").write_text("rq1 transitions")
    (rq1 / "model.zip").write_text("model")
    for name in ("in-place", "atomic"):
        shutil.copytree(rq1, tmp_path / name, copy_function=link_replay_buffer)
    assert os.path.samefile(tmp_path / "atomic" / "replay_buffer.pkl", rq1 / "replay_buffer.pkl")
    assert not os.path.samefile(tmp_path / "atomic" / "model.zip", rq1 / "model.zip")

    (tmp_path / "in-place" / "replay_buffer.pkl").write_text("repair transitions")
    assert (rq1 / "replay_buffer.pkl").read_text() == "repair transitions"
    (rq1 / "replay_buffer.pkl").unlink()
    (rq1 / "replay_buffer.pkl").write_text("rq1 transitions")

    shutil.rmtree(tmp_path / "atomic")
    shutil.copytree(rq1, tmp_path / "atomic", copy_function=link_replay_buffer)
    atomic_write_text(tmp_path / "atomic" / "replay_buffer.pkl", "repair transitions")
    assert (rq1 / "replay_buffer.pkl").read_text() == "rq1 transitions"
