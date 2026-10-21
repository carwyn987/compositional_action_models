"""scripts/overnight_runs.py: the experiment's settings, the repairs it accepts, and the RQ1 budget set from the
learnability runs (RQ1_STEPS=auto)."""

import json
import os
import re
import subprocess
import sys

import pytest

from cam.experiments.research_experiments import EXPERIMENTS
from cam.skills.registry import SKILL_REGISTRY
from scripts import overnight_runs


@pytest.mark.unit
def test_rq1_budget_is_the_margin_times_what_each_skill_needed_alone():
    """pickup 100k + putdown 50k + unstack (solved at step 0: one evaluation interval, 50k) + stack (never: all of
    its 150k) = 350k; x 1.5 = 525k, rounded up to whole 50k intervals: 550k."""
    steps = {"pickup": 100_000, "putdown": 50_000, "unstack": 0, "stack": None}
    assert overnight_runs.rq1_budget(steps, margin=1.5, learn_steps=150_000, interval=50_000) == 550_000


@pytest.mark.unit
def test_learnability_steps_read_each_skills_result_and_name_the_missing(tmp_path, monkeypatch):
    """Steps to threshold come from each skill's learnability metrics.json; a skill without one is reported."""
    monkeypatch.setattr(overnight_runs, "OUT", tmp_path)
    for skill, steps in (("pickup", 100_000), ("stack", None)):
        run = tmp_path / "learnability" / skill / f"{skill}_run"
        run.mkdir(parents=True)
        summary = {"deterministic": {skill: {"steps_to_threshold": steps}}}
        (run / "metrics.json").write_text(json.dumps({"summary": summary}))
    steps, missing = overnight_runs.learnability_steps(["pickup", "stack", "putdown"])
    assert steps == {"pickup": 100_000, "stack": None, "putdown": None} and missing == ["putdown"]


@pytest.mark.unit
@pytest.mark.parametrize("auto", [False, True])
def test_rq1_runs_wait_for_the_learnability_runs_only_with_an_automatic_budget(monkeypatch, auto):
    """Control: with a fixed RQ1_STEPS the rq1 runs start without waiting. With RQ1_STEPS=auto each rq1 run waits for
    the learnability run of every base skill, whose results set its budget."""
    for name, value in (("RUN_GROUPS", ["learnability", "rq1"]), ("OPERATOR_EMBEDDINGS", ["one-hot"]), ("SEEDS", [0]),
                        ("BASE_SKILLS", ["pickup", "putdown"]), ("REPAIRS", []), ("REPAIRED", []), ("RQ1_AUTO", auto)):
        monkeypatch.setattr(overnight_runs, name, value)
    (rq1,) = [run for run in overnight_runs.build_plan() if run.group == "rq1"]
    assert rq1.depends_on == (("learnability_pickup", "learnability_putdown") if auto else ())


@pytest.mark.unit
def test_the_budget_goes_into_every_rq1_runs_command(tmp_path, monkeypatch):
    """Once the learnability results exist, each rq1 run trains for the budget: its steps and --total-timesteps."""
    monkeypatch.setattr(overnight_runs, "OUT", tmp_path)
    monkeypatch.setattr(overnight_runs, "BASE_SKILLS", ["pickup", "putdown"])
    for skill, steps in (("pickup", 100_000), ("putdown", 50_000)):
        run = tmp_path / "learnability" / skill / "run"
        run.mkdir(parents=True)
        (run / "metrics.json").write_text(json.dumps({"summary": {"deterministic": {skill: {"steps_to_threshold": steps}}}}))
    runs = [overnight_runs.Run(f"rq1_{condition}", "rq1", "", "", "", 1, condition, tmp_path,
                               ["main.py", "--total-timesteps", "1"]) for condition in ("one-hot", "random")]
    runner = overnight_runs.Runner(runs)
    for run in runs:
        runner.set_rq1_budget(run)
    expected = overnight_runs.rq1_budget({"pickup": 100_000, "putdown": 50_000}, overnight_runs.RQ1_BUDGET_MARGIN,
                                         overnight_runs.LEARN_STEPS, overnight_runs.LEARN_EVAL_INTERVAL)
    assert [(run.steps, run.command[-1]) for run in runs] == [(expected, str(expected))] * 2


@pytest.mark.unit
def test_the_plan_runs_learnability_then_every_rq1_run_then_the_repairs(monkeypatch):
    """Runs start in plan order, so the most informative come first: whether each skill can be learned, then every
    condition's rq1 result, then the repairs (which wait for their rq1 run anyway)."""
    for name, value in (("RUN_GROUPS", ["learnability", "rq1", "repair"]), ("OPERATOR_EMBEDDINGS", ["one-hot", "text"]),
                        ("SEEDS", [0]), ("BASE_SKILLS", ["pickup", "putdown", "stack"]),
                        ("REPAIRS", [("stack", "place-beside")]), ("REPAIRED", ["place-beside"]),
                        ("REPAIR_MODES", ["isolated"]), ("REPAIR_HIDDEN_EMBEDDINGS", ["one-hot"]), ("RQ1_AUTO", False)):
        monkeypatch.setattr(overnight_runs, name, value)
    groups = [run.group for run in overnight_runs.build_plan()]
    assert groups == ["learnability"] * 4 + ["rq1"] * 2 + ["repair"] * 3  # one-hot: visible and hidden; text: visible


@pytest.mark.unit
def test_a_repair_asking_for_an_untrained_literal_stops_the_runner(monkeypatch):
    """Control: with push-beside among the base skills, stack -> place-beside passes. Without it, place-beside's
    (beside ?o ?u) is in no base skill, and the runner stops before anything runs, naming the skill that has it."""
    for name, value in (("RUN_GROUPS", ["learnability", "rq1", "repair"]), ("REPAIRS", [("stack", "place-beside")]),
                        ("BASE_SKILLS", ["pickup", "putdown", "stack", "push-beside"])):
        monkeypatch.setattr(overnight_runs, name, value)
    overnight_runs.check_repairs()
    monkeypatch.setattr(overnight_runs, "BASE_SKILLS", ["pickup", "putdown", "stack"])
    with pytest.raises(SystemExit, match=r"no base skill has effect \(beside \?o \?u\) \(add one of: push-beside\)"):
        overnight_runs.check_repairs()


@pytest.mark.unit
def test_settings_come_from_the_experiment_unless_set_in_the_environment(monkeypatch):
    """The experiment's settings in the runner's format; an environment variable replaces one and is reported as an
    override; machine settings (JOBS) are not the experiment's."""
    monkeypatch.setattr(overnight_runs, "EXPERIMENT", EXPERIMENTS["rq3-repair-minimal"])
    monkeypatch.setattr(overnight_runs, "OVERRIDES", {})
    for name in ("OPERATOR_EMBEDDINGS", "REPAIRS", "JOBS"):
        monkeypatch.delenv(name, raising=False)
    assert overnight_runs.setting("REPAIRS") == "unstack:unstack-raised putdown:place-beside push-left:push-left-front"
    assert overnight_runs.setting("OPERATOR_EMBEDDINGS") == "compositional-tree one-hot multi-hot-trainable text"
    monkeypatch.setenv("OPERATOR_EMBEDDINGS", "one-hot")
    monkeypatch.setenv("JOBS", "3")
    assert overnight_runs.setting("OPERATOR_EMBEDDINGS") == "one-hot" and overnight_runs.setting("JOBS", "2") == "3"
    assert overnight_runs.OVERRIDES == {"OPERATOR_EMBEDDINGS": "one-hot"}


@pytest.mark.unit
def test_the_command_line_chooses_the_experiment(tmp_path):
    """--experiment learnability plans one run per registered skill, alone; the default plans rq3-repair-minimal's runs."""
    environment = {name: value for name, value in os.environ.items() if name not in EXPERIMENTS["rq3-repair-minimal"].settings()}
    environment |= {"DRY_RUN": "1", "OUT": str(tmp_path)}

    def plan(*arguments: str) -> list[str]:
        output = subprocess.run([sys.executable, "scripts/overnight_runs.py", *arguments], cwd=overnight_runs.ROOT,
                                env=environment, capture_output=True, text=True, check=True).stdout
        return re.findall(r"^\s+\d+\. (\S+)", output, re.MULTILINE)

    assert plan("--experiment", "learnability") == ["Learnability"] * len(SKILL_REGISTRY)
    static = subprocess.run([sys.executable, "scripts/overnight_runs.py", "--experiment", "overlap"],
                            cwd=overnight_runs.ROOT, env=environment, capture_output=True, text=True, check=True).stdout
    assert "Representation overlap: 13 training skills" in static and "Checks before starting" not in static
    assert set(plan()) == {"Learnability", "RQ1", "Repair"}


@pytest.mark.unit
def test_the_review_reports_a_repairs_transfer_gain_against_its_learnability_run(tmp_path):
    """pickup-raised needed 60k steps alone (its learnability run) and 20k on itself after being added to a repaired
    run: a transfer gain of 3x. A repair without a learnability run in the folder (the control) reports none."""
    from scripts.review_runs import section_added_skills

    def run(path, points, summary):
        path.mkdir(parents=True)
        (path / "metrics.json").write_text(json.dumps({"curves": {"deterministic": {"points": points}},
                                                       "summary": summary}))
        return path

    def summary(skill, **values):
        return {mode: {skill: values} for mode in ("deterministic", "stochastic")}

    alone = run(tmp_path / "learnability" / "pickup-raised" / "run", [{"step": 0, "result": {"pickup-raised": {}}}],
                summary("pickup-raised", steps_to_threshold=60_000))
    repaired = run(tmp_path / "repair" / "pickup-raised" / "tree", [
        {"step": 0, "result": {"pickup": {}}}, {"step": 500_000, "result": {"pickup-raised": {}}}],
        summary("pickup-raised", zero_shot_success_rate=0.0, steps_to_threshold=20_000, skill_steps_to_threshold=20_000))
    assert any("transfer gain 3.00x" in line for line in section_added_skills([alone, repaired]))
    assert not any("transfer gain" in line for line in section_added_skills([repaired]))


@pytest.mark.unit
def test_progress_is_read_from_the_runs_progress_file(tmp_path, monkeypatch):
    """A run's progress comes from progress.json in its directory; before training has written one (the control),
    it is unknown."""
    monkeypatch.setattr(overnight_runs, "OUT", tmp_path)
    run = overnight_runs.Run("rq1_one-hot_seed0", "rq1", "", "", "", 300_000, "one-hot", tmp_path / "rq1", ["main.py"])
    directory = tmp_path / "rq1" / "pickup_mtsac_one-hot_shaped_seed0"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    assert overnight_runs.Runner.progress(run) == (None, 300_000)
    (directory / "progress.json").write_text(json.dumps({"done": 172_987, "total": 300_000, "step": 172_987}))
    assert overnight_runs.Runner.progress(run) == (172_987, 300_000)