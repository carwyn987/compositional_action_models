"""main.py run lifecycle: new run, resume with an added skill, evaluate only, locked options."""

import json

import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import SAC  # noqa: E402

from main import main, parse_args, run_name  # noqa: E402

FAST = ["--algorithm", "sac", "--num-blocks", "2", "--max-steps-per-episode", "10", "--eval-episodes", "1"]


def run_directory(tmp_path, skills):
    return tmp_path / run_name(parse_args(["--skills", *skills, *FAST]))


@pytest.mark.integration
def test_new_run_then_resume_with_added_skill_then_evaluate(tmp_path):
    """Train pickup 200 steps; resume adding putdown for 100 more (its zero-shot point is the resume step);
    then evaluate only. Curves, step count, monitor.csv and train.log continue in the same directory."""
    main(["--skills", "pickup", *FAST, "--total-timesteps", "200", "--eval-interval", "100", "--output-dir", str(tmp_path)])
    run = run_directory(tmp_path, ["pickup"])
    main(["--run-directory", str(run), "--skills", "pickup", "putdown", "--total-timesteps", "100"])

    assert SAC.load(run / "model.zip").num_timesteps == 300
    metrics = json.loads((run / "metrics.json").read_text())
    points = metrics["curves"]["deterministic"]["points"]
    assert [point["step"] for point in points] == [0, 100, 200, 300]
    assert "putdown" not in points[1]["result"] and "putdown" in points[2]["result"]  # added at step 200
    assert metrics["progress"]["steps"] == 300
    assert json.loads((run / "config.json").read_text())["skills"] == ["pickup", "putdown"]
    assert (run / "train.log").read_text().count("to resume or continue this run") == 2

    main(["--mode", "evaluate", "--run-directory", str(run), "--eval-skills", "putdown"])
    (evaluation_file,) = run.glob("evaluation_*.json")
    evaluation = json.loads(evaluation_file.read_text())
    assert evaluation["step"] == 300 and set(evaluation["results"]["deterministic"]) == {"putdown"}


@pytest.mark.unit
def test_resume_rejects_options_that_change_the_observation(tmp_path):
    """Options defining the observation or algorithm (e.g. --operator-encoder) cannot change on resume."""
    (tmp_path / "config.json").write_text(json.dumps(parse_args(["--skills", "pickup"])))
    with pytest.raises(SystemExit):
        parse_args(["--run-directory", str(tmp_path), "--operator-encoder", "one-hot"])
    assert parse_args(["--run-directory", str(tmp_path), "--total-timesteps", "5"])["total_timesteps"] == 5
