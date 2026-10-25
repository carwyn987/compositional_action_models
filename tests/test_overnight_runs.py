"""scripts/overnight_runs.py: the experiment's settings, the repairs it accepts, the RQ1 budget set from the
learnability runs (RQ1_STEPS=auto), and the rq2 chains: their orders, stages and when a chain ends."""

import json
import os
import re
import subprocess
import sys
from itertools import permutations

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
@pytest.mark.parametrize("names, conditions", [
    (["exclude"], ["one-hot", "text-no-name", "compositional-tree-no-name"]),
    (["include"], ["one-hot", "text", "compositional-tree"]),
    (["exclude", "include"], ["one-hot", "text-no-name", "text", "compositional-tree-no-name", "compositional-tree"]),
])
def test_operator_names_choose_the_options_and_labels_of_the_conditions_with_a_name(monkeypatch, names, conditions):
    """Each OPERATOR_NAMES value is a variant of the conditions whose embedding carries the operator's name: exclude
    leaves it out of main.py's config (text and compositional alike) and labels the runs <condition>-no-name, include
    keeps main.py's default and the condition's label. One-hot has no name (the control): one run, whatever the
    setting. Each repair continues the rq1 run of its own variant, and no two runs share main.py's run name (the
    rows of summary.csv)."""
    from main import parse_args, run_name

    for name, value in (("RUN_GROUPS", ["rq1", "repair"]), ("OPERATOR_NAMES", names),
                        ("OPERATOR_EMBEDDINGS", ["one-hot", "text", "compositional-tree"]), ("SEEDS", [0]),
                        ("BASE_SKILLS", ["pickup", "unstack", "pickup-raised"]),
                        ("REPAIRS", [("unstack", "unstack-raised")]), ("REPAIRED", ["unstack-raised"]),
                        ("REPAIR_MODES", ["isolated"]), ("REPAIR_HIDDEN_EMBEDDINGS", []), ("RQ1_AUTO", False)):
        monkeypatch.setattr(overnight_runs, name, value)
    plan = overnight_runs.build_plan()
    rq1 = {run.label.removeprefix("rq1_").removesuffix("_seed0"): run for run in plan if run.group == "rq1"}
    assert list(rq1) == conditions
    for condition, run in rq1.items():
        config = parse_args(run.command[1:])
        excluded = condition.endswith("-no-name")
        assert (config["operator_text_name"], config["compositional_name"]) == (
            "none" if excluded and config["operator_encoder"] == "text" else "include",
            "none" if excluded and config["operator_encoder"] == "compositional" else "text")
        assert run.output == overnight_runs.OUT / "rq1" / f"{condition}_seed0"
        (repair,) = [repair for repair in plan if repair.copy_from == run.output]
        assert repair.label == f"repair_unstack-raised_{condition}_repair-visible_isolated_seed0"
        assert repair.embedding_options == run.embedding_options
        assert " ".join(run.embedding_options) in " ".join(run.command)
    assert len({run_name(parse_args(run.command[1:])) for run in rq1.values()}) == len(conditions)


@pytest.mark.unit
def test_the_preflight_embeds_the_texts_the_runs_use(monkeypatch):
    """With the name excluded, the text runs' paragraphs are embedded without it, and no name is. With it included
    (the control), each paragraph names its operator, and the compositional runs' names are embedded too."""
    from cam.experiments import environment_setup
    from cam.representations.text_backends import MockTextBackend

    embedded = []

    class RecordingBackend(MockTextBackend):
        def embed_text(self, text):
            embedded.append(text)
            return super().embed_text(text)

    monkeypatch.setattr(environment_setup, "text_backend",
                        lambda config, dim=None: RecordingBackend(dim or config["operator_embedding_dim"]))
    for name, value in (("RUN_GROUPS", ["rq1"]), ("OPERATOR_EMBEDDINGS", ["text", "compositional-tree"]),
                        ("SEEDS", [0]), ("BASE_SKILLS", ["pickup", "pickup-raised"]), ("TEXT_BACKEND", "mock"),
                        ("RQ1_AUTO", False)):
        monkeypatch.setattr(overnight_runs, name, value)
    texts = {}
    for choice in ("exclude", "include"):
        monkeypatch.setattr(overnight_runs, "OPERATOR_NAMES", [choice])
        embedded.clear()
        overnight_runs.check_text_backend(["pickup", "pickup-raised"],
                                          {run.embedding_options for run in overnight_runs.build_plan()})
        texts[choice] = list(embedded)
    assert len(texts["exclude"]) == 2 and not any("pickup" in text for text in texts["exclude"])
    assert all(text.startswith("Action unnamed with parameters") for text in texts["exclude"])
    assert [text.split(" with")[0] for text in texts["include"][:2]] == ["Action pickup", "Action pickup-raised"]
    assert texts["include"][2:] == ["pickup", "pickup-raised"]


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


def option(command: list[str], flag: str) -> list[str]:
    """The values a command gives its last flag (up to the next option)."""
    start = len(command) - command[::-1].index(flag)
    end = next((index for index in range(start, len(command)) if command[index].startswith("--")), len(command))
    return command[start:end]


def plan_rq2(monkeypatch, tmp_path, order: str, embeddings=("one-hot",), seeds=(0,)) -> list:
    for name, value in (("OUT", tmp_path), ("RUN_GROUPS", ["rq2"]), ("OPERATOR_EMBEDDINGS", list(embeddings)),
                        ("SEEDS", list(seeds)), ("RQ2_ORDERS", [overnight_runs.order_stages(order)])):
        monkeypatch.setattr(overnight_runs, name, value)
    return overnight_runs.build_plan()


@pytest.mark.unit
def test_an_order_is_stages_of_skills_and_the_runner_checks_it(monkeypatch):
    """RQ2_ORDERS="pickup+unstack,unstack-raised": pickup and unstack together, then unstack-raised, and the order's
    name is the same text. Control: an unknown skill, or one in two stages, stops the runner before anything runs."""
    order = overnight_runs.order_stages("pickup+unstack,unstack-raised")
    assert order == [["pickup", "unstack"], ["unstack-raised"]]
    assert overnight_runs.order_name(order) == "pickup+unstack,unstack-raised"
    monkeypatch.setattr(overnight_runs, "RUN_GROUPS", ["rq2"])
    monkeypatch.setattr(overnight_runs, "RQ2_ORDERS", [order])
    overnight_runs.check_orders()
    for written, problem in (("pickup,unstak", r"unknown skills \['unstak'\]"),
                             ("pickup+unstack,pickup", "pickup in more than one stage")):
        monkeypatch.setattr(overnight_runs, "RQ2_ORDERS", [overnight_runs.order_stages(written)])
        with pytest.raises(SystemExit, match=problem):
            overnight_runs.check_orders()


@pytest.mark.unit
def test_shuffled_orders_are_different_and_the_same_every_time():
    """Each shuffle has every skill, one per stage, and differs from the others; the fixed seed gives the same orders
    every time, and asking for more keeps the first ones. At most every order there is: 6 of 3 skills."""
    skills = list(EXPERIMENTS["rq2"].base_skills)
    orders = overnight_runs.shuffled_orders(skills, 3)
    assert orders == overnight_runs.shuffled_orders(skills, 3) == overnight_runs.shuffled_orders(skills, 5)[:3]
    assert all(len(stage) == 1 for order in orders for stage in order)
    assert all(sorted(skill for (skill,) in order) == sorted(skills) for order in orders)
    assert len({overnight_runs.order_name(order) for order in orders}) == 3
    assert len(overnight_runs.shuffled_orders(skills[:3], 10)) == 6
    assert [order[-1] for order in orders] == [["pickup-raised"], ["pickup"], ["unstack"]]  # as rq2's docstring says


@pytest.mark.unit
def test_in_every_order_of_rq2s_skills_the_last_one_added_is_covered():
    """rq2's docstring: over the 24 orders of its four skills, the fourth skill added has literal coverage 1 by the
    three before it every time, the third in the 8 orders that start with opposite corners, the second never."""
    orders = [[[skill] for skill in order] for order in permutations(EXPERIMENTS["rq2"].base_skills)]
    covered = {number: sum(overnight_runs.stage_coverage(order, number)[order[number - 1][0]] == 1 for order in orders)
               for number in (2, 3, 4)}
    assert len(orders) == 24 and covered == {2: 0, 3: 8, 4: 24}


@pytest.mark.unit
def test_an_rq2_chain_is_one_run_continued_stage_by_stage(tmp_path, monkeypatch):
    """Per seed, order and embedding: stage 1 is a new run on its skills, stopping once each is at the threshold and
    saving its replay buffer; each later stage waits for the one before and continues that run with its skills added,
    stopping on them in repair mode retain."""
    runs = plan_rq2(monkeypatch, tmp_path, "pickup+unstack,unstack-raised,pickup-raised", ("one-hot", "text"), (0, 1))
    assert len(runs) == 2 * 2 * 3 and {run.group for run in runs} == {"rq2"}
    first, second, third = runs[:3]
    chain = "rq2_pickup+unstack,unstack-raised,pickup-raised"
    assert [run.label for run in runs[::3]] == [f"{chain}_{condition}_seed{seed}_stage1"
                                                for seed in (0, 1) for condition in ("one-hot", "text-no-name")]
    assert [run.label for run in runs[:3]] == [f"{chain}_one-hot_seed0_stage{number}" for number in (1, 2, 3)]
    assert [run.depends_on for run in runs[:3]] == [(), (first.label,), (second.label,)]
    assert first.output == second.output == third.output == (
        tmp_path / "rq2" / "one-hot_seed0" / "pickup+unstack,unstack-raised,pickup-raised")

    assert option(first.command, "--skills") == option(first.command, "--stop-skills") == ["pickup", "unstack"]
    assert "--save-replay-buffer" in first.command and "--repair-mode" not in first.command
    assert option(first.command, "--output-dir") == [str(first.output)]
    assert option(first.command, "--total-timesteps") == [str(overnight_runs.RQ2_FIRST_STEPS)]
    assert option(first.command, "--eval-interval") == [str(overnight_runs.RQ2_EVAL_INTERVAL)]
    for run, skills in ((second, ["pickup", "unstack", "unstack-raised"]),
                        (third, ["pickup", "unstack", "unstack-raised", "pickup-raised"])):
        assert option(run.command, "--run-directory") == [str(first.output)]  # where stage 1 makes its run
        assert option(run.command, "--skills") == skills
        assert option(run.command, "--stop-skills") == list(run.added) == skills[-1:]
        assert option(run.command, "--repair-mode") == ["retain"]
        assert option(run.command, "--total-timesteps") == [str(overnight_runs.RQ2_STAGE_STEPS)]


def make_run(directory, steps: int, points: list[dict]):
    """A run directory as main.py leaves it, at the given step and with the given evaluation points."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}")
    (directory / "model.zip").write_text("model")
    (directory / "metrics.json").write_text(json.dumps({"progress": {"steps": steps},
                                                        "curves": {"deterministic": {"points": points}}}))
    return directory


@pytest.mark.unit
def test_a_chain_ends_at_a_stage_that_did_not_reach_the_threshold(tmp_path, monkeypatch):
    """Stage 1 ran its whole budget without reaching the threshold: stage 2 does not start, the chain ends with the
    reason (also in the run's chain_ended.txt, for review_runs.py), and stage 3 is skipped. Control: once stage 1's
    log says it stopped at the threshold, stage 2 starts, continuing stage 1's run."""
    first, second, third = plan_rq2(monkeypatch, tmp_path, "pickup,unstack,pickup-raised")
    run = make_run(first.output / "pickup_mtsac_one-hot_shaped_seed0", 500_000, [{"step": 0, "result": {"pickup": {}}}])
    started = []
    monkeypatch.setattr(overnight_runs.subprocess, "Popen", lambda command, **_: started.append(command))
    first.log.parent.mkdir()

    def start_stages_after(log: str) -> overnight_runs.Runner:
        first.log.write_text(log)
        runner = overnight_runs.Runner([first, second, third])
        runner.done.add(first.label)
        runner.start(second, 2)
        if second.label in runner.failed:
            runner.start(third, 3)
        return runner

    runner = start_stages_after("training finished at step 500000\n")
    assert started == [] and runner.ended == {second.label, third.label} <= runner.failed
    assert (run / "chain_ended.txt").read_text() == (
        f"chain ended: stage 1 did not reach the success threshold (its {overnight_runs.RQ2_FIRST_STEPS:,} steps ran "
        "out), so stage 2 and any after it did not run\n")

    start_stages_after("stopping training at step 120000: pickup reached success rate 0.95\n"
                       "training finished at step 120000\n")
    (command,) = started
    assert option(command, "--run-directory") == [str(run)] and option(command, "--stop-skills") == ["unstack"]
    assert not (run / "chain_ended.txt").exists()


@pytest.mark.unit
def test_an_interrupted_stage_resumes_its_run_with_the_steps_it_has_left(tmp_path, monkeypatch):
    """Stage 2 added unstack at step 120k (its first evaluation) and was interrupted at 170k: it resumes the same run
    with its budget less the 50k done. Control: before stage 2 starts (unstack never evaluated), its whole budget."""
    first, second = plan_rq2(monkeypatch, tmp_path, "pickup,unstack")
    directory = first.output / "pickup_mtsac_one-hot_shaped_seed0"
    before = [{"step": 0, "result": {"pickup": {}}}, {"step": 120_000, "result": {"pickup": {}}}]
    make_run(directory, 120_000, before)
    runner = overnight_runs.Runner([first, second])
    command = runner.prepare(second)
    assert option(command, "--run-directory") == [str(directory)]
    assert option(command, "--total-timesteps") == [str(overnight_runs.RQ2_STAGE_STEPS)]
    make_run(directory, 170_000, [*before[:1], {"step": 120_000, "result": {"pickup": {}, "unstack": {}}}])
    command = runner.prepare(second)
    assert option(command, "--run-directory") == [str(directory)]
    assert option(command, "--total-timesteps") == [str(overnight_runs.RQ2_STAGE_STEPS - 50_000)]


@pytest.mark.unit
def test_rq2_add_unstack_raised_plans_one_chain_of_two_stages_per_embedding(tmp_path):
    """Per operator embedding (one-hot, compositional-tree without the operator's name): a new run on pickup, unstack and pickup-raised until each
    reaches the threshold (at most 1.5M steps, evaluated every 10k, saved every 20k with its replay buffer), then that
    run with unstack-raised added, training every skill (retain) until it is at the threshold and the others are back
    at their level, at most 1M steps. Every literal of unstack-raised is in the three: coverage 1."""
    experiment = EXPERIMENTS["rq2-add-unstack-raised"]
    environment = {name: value for name, value in os.environ.items() if name not in experiment.settings()}
    environment |= {"DRY_RUN": "1", "OUT": str(tmp_path)}
    output = subprocess.run([sys.executable, "scripts/overnight_runs.py", "--experiment", experiment.name],
                            cwd=overnight_runs.ROOT, env=environment, capture_output=True, text=True, check=True).stdout
    commands = {label: command.split() for label, command in re.findall(r"^(\S+):\n  \S+ (main\.py .*)$", output,
                                                                        re.MULTILINE)}
    chain = "rq2_pickup+unstack+pickup-raised,unstack-raised"
    assert list(commands) == [f"{chain}_{condition}_seed0_stage{number}"
                              for condition in ("one-hot", "compositional-tree-no-name") for number in (1, 2)]
    first, added = commands[f"{chain}_one-hot_seed0_stage1"], commands[f"{chain}_one-hot_seed0_stage2"]
    assert option(first, "--skills") == option(first, "--stop-skills") == ["pickup", "unstack", "pickup-raised"]
    assert [option(first, flag) for flag in ("--total-timesteps", "--eval-interval", "--save-interval")] == [
        ["1500000"], ["10000"], ["20000"]]
    assert "--save-replay-buffer" in first and option(first, "--algorithm") == ["mtsac"]
    assert option(added, "--skills") == ["pickup", "unstack", "pickup-raised", "unstack-raised"]
    assert option(added, "--stop-skills") == ["unstack-raised"] and option(added, "--repair-mode") == ["retain"]
    assert option(added, "--total-timesteps") == ["1000000"]
    assert "pickup+unstack+pickup-raised,unstack-raised: unstack-raised 1.00" in output


@pytest.mark.unit
def test_the_review_reports_every_addition_of_a_chain_with_its_own_retention(tmp_path):
    """An rq2 chain's run added unstack at step 100 and pickup-raised at 300: section 5 reports each where it was added,
    in that order, with what the other skills forgot after it (unstack's from earlier_retention). Control: without
    earlier_retention, unstack gets no retention rather than pickup-raised's."""
    from scripts.review_runs import section_added_skills

    def retention(added_at: int, lowest: float) -> dict:
        others = {"level_before": 1.0, "lowest_after": lowest, "final": 1.0, "recovered_at": None, "recovered": True}
        return {"added_at": added_at, "skills": {"pickup": others}, "all_recovered_at": added_at + 100}

    run = tmp_path / "rq2" / "one-hot_seed0" / "pickup,unstack,pickup-raised" / "pickup_mtsac_one-hot_shaped_seed0"
    run.mkdir(parents=True)
    points = [{"step": 0, "result": {"pickup": {}}}, {"step": 100, "result": {"pickup": {}, "unstack": {}}},
              {"step": 300, "result": {"pickup": {}, "unstack": {}, "pickup-raised": {}}}]
    added = {"zero_shot_success_rate": 0.0, "steps_to_threshold": 50, "skill_steps_to_threshold": 30}
    summary = {mode: {"unstack": added, "pickup-raised": added} for mode in ("deterministic", "stochastic")}

    def entries(**metrics) -> list[str]:
        (run / "metrics.json").write_text(json.dumps({"curves": {"deterministic": {"points": points}},
                                                      "summary": summary, **metrics}))
        return "\n".join(section_added_skills([run])).split("\n   pickup,unstack,pickup-raised/")[1:]

    unstack, raised = entries(retention=retention(300, 0.6), earlier_retention=[retention(100, 0.2)])
    assert unstack.startswith("pickup_mtsac_one-hot_shaped unstack  (1 seeds; added at step 100)")
    assert raised.startswith("pickup_mtsac_one-hot_shaped pickup-raised  (1 seeds; added at step 300)")
    assert re.search(r"lowest after\s+0.2 ", unstack) and re.search(r"lowest after\s+0.6 ", raised)
    unstack, raised = entries(retention=retention(300, 0.6))
    assert "lowest after" not in unstack and re.search(r"lowest after\s+0.6 ", raised)


@pytest.mark.unit
def test_an_interrupted_repair_starts_over_from_its_rq1_run(tmp_path, monkeypatch):
    """Resumed, an interrupted repair would reload its rq1 run's replay buffer with the replaced skill's transitions
    back in, and its config.json would no longer show the replacement; so it starts over from a fresh copy of its rq1
    run, with its whole step budget. Control: a repair not started yet is copied the same way."""
    monkeypatch.setattr(overnight_runs, "OUT", tmp_path)
    rq1 = tmp_path / "rq1" / "one-hot_seed0" / "run"
    rq1.mkdir(parents=True)
    for name in ("config.json", "model.zip", "metrics.json"):
        (rq1 / name).write_text("{}")
    (rq1 / "replay_buffer.pkl").write_text("the rq1 run's transitions")
    repair = tmp_path / "repair" / "unstack-raised" / "one-hot_repair-visible_isolated_seed0"
    command = ["main.py", "--run-directory", str(repair), "--total-timesteps", "150000"]
    run = overnight_runs.Run("repair_unstack-raised", "repair", "", "", "", 150_000, "one-hot", repair, command,
                             copy_from=rq1.parent)
    runner = overnight_runs.Runner([run])
    assert runner.prepare(run) == command and (repair / "replay_buffer.pkl").samefile(rq1 / "replay_buffer.pkl")
    (repair / "model.zip").write_text("saved after some of the repair's training")  # then interrupted
    assert runner.prepare(run) == command
    assert (repair / "model.zip").read_text() == "{}" and (repair / "replay_buffer.pkl").samefile(rq1 / "replay_buffer.pkl")
