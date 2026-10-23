#!/usr/bin/env python3
"""Overnight experiments for the research questions: an experiment's plan of training runs, executed and explained.

    ./scripts/start_overnight.sh --experiment rq3-repair-minimal       # usual way: environment checks + unit tests, then this
    .venv/bin/python scripts/overnight_runs.py --list          # the experiments: what each asks, and its settings
    .venv/bin/python scripts/overnight_runs.py --experiment rq1                      # this runner alone
    DRY_RUN=1 .venv/bin/python scripts/overnight_runs.py --experiment rq1            # print the plan, run nothing
    SEEDS="0 1" JOBS=6 .venv/bin/python scripts/overnight_runs.py --experiment learnability

Experiments (--experiment NAME, default rq3-repair-minimal) are defined in src/cam/experiments/research_experiments.py:
learnability (each skill alone), rq1 (one shared policy per operator embedding), rq2 and rq2-add-unstack-raised
(skills added to a shared policy a stage at a time), rq3-repair-minimal and rq3-repair-maximal (rq1's policies
continued with repaired operators), and overlap (static: how much of their operators the training skills share).
Each sets the groups to run and every setting below; an environment variable of a setting's name overrides it, and
the runner lists the overrides when it starts. A plan whose repairs ask for a literal that no training skill has
refuses to start.

How it runs
  * Up to JOBS runs at once (default 2). One run uses a fraction of a GPU or CPU: SAC alternates a
    simulator step (CPU) with one small gradient step, so several runs side by side get through the plan
    several times faster. JOBS=1 runs one at a time.
  * Runs start in plan order, most informative first: every learnability run (can each skill be learned at
    all?), then every rq1 run (each condition and seed), then the repairs, then the rq2 stages.
  * Each run is a child process (main.py) whose full output goes to <OUT>/logs/<label>.log. The terminal
    shows, as each run starts, what it trains and why (which research question or check it serves);
    while runs go, their progress (steps, steps/s, time left) and the overall time left; when one ends,
    its outcome. A repair run waits for the RQ1 run it continues, an rq2 stage for the stage before.
  * Ctrl+C stops every running run after it saves. Starting the same command again skips finished runs and
    resumes interrupted ones from their last save (an interrupted repair starts over from its rq1 run).
  * Before anything long starts (fail fast): the text backend is checked and every operator text the runs
    need is embedded and cached (TEXT_BACKEND=openai needs the API key and network), the device is reported,
    and every skill's scripted policy must succeed at least once in 3 episodes (which checks each skill's
    setup and predicates).

Groups
  learnability  each skill alone (BASE_SKILLS and every repaired skill), LEARNABILITY_EMBEDDING: can it be trained at
                all, and how fast? Also the "individual policy" baseline for RQ4 (hypernetworks).
  rq1           RQ1: one shared policy on BASE_SKILLS for each operator embedding condition (OPERATOR_EMBEDDINGS)
                and seed: steps to threshold, AUC and final success per condition. Conditions whose embedding
                carries the operator's name (text, compositional-*) run once per OPERATOR_NAMES: exclude, as
                <condition>-no-name, and include, as <condition>.
  repair        RQ3: for each FROM:TO in REPAIRS, each rq1 run continued with FROM replaced by its
                repaired version TO: TO's zero-shot success at the resume step, then the steps it needs to
                reach the success threshold (TO evaluated alone 1k, 2k, 4k, ... steps after it is added,
                then every REPAIR_EVAL_INTERVAL), at most REPAIR_STEPS. Each REPAIR_MODES (main.py
                --repair-mode), and each of two variants:
                  isolated        trains only TO; the other skills are evaluated, not trained, which shows
                                  what they forget; stops once TO is at the threshold
                  retain          trains every skill, as the rq1 run did; stops once TO is at the threshold
                                  and every other skill is back at its level before the repair
                  repair-visible  the repaired operator is encoded on its own: content-based embeddings
                                  (multi-hot, text, compositional) show the changed literals; one-hot and
                                  random give it a code no training has used
                  repair-hidden   (REPAIR_HIDDEN_EMBEDDINGS only) the repaired operator gets exactly FROM's
                                  embedding (--operator-identity-aliases), so the policy cannot see the
                                  repair. A control: if it succeeds zero-shot, the repair needs no new
                                  behaviour and the test is invalid (review.txt flags it).
  rq2           RQ2: for each order (RQ2_ORDERS, and RQ2_SHUFFLES shuffles of BASE_SKILLS), operator embedding and
                seed, a chain of stages in one run. Stage 1: a new run on its skills until each reaches the
                success threshold, at most RQ2_FIRST_STEPS. Each later stage: the same run continued with its
                skills added, evaluated zero-shot and densely after (as a repair's TO), training every skill
                (retain) until they are at the threshold and the earlier skills back at their level, at most
                RQ2_STAGE_STEPS. A stage starts only if the one before stopped at its threshold; otherwise the
                chain ends there, with the reason. The plan gives each added skill's literal coverage by the
                skills before it: zero-shot success needs 1. Every skill evaluated every RQ2_EVAL_INTERVAL steps.

Outputs, under OUT (default outputs/overnight_<date>)
  learnability/<skill>/<run name>/, rq1/<condition>_seed<n>/<run name>/   run directories (metrics.json,
                                                                            model.zip, tensorboard/, ...)
  repair/<TO>/<condition>_<repair-visible|repair-hidden>_seed<n>/          copy of the rq1 run, continued
  rq2/<condition>_seed<n>/<order>/<run name>/                              one run per chain (chain_ended.txt:
                                                                            why it ended before its last stage)
  logs/<label>.log   each run's full output        failed.txt   labels of runs that failed
  summary.csv        per run and skill metrics     review.txt   scripts/review_runs.py's report
View the curves with: LOGDIR=<OUT> ./scripts/start_tensorboard.sh

Settings: the experiment's (research_experiments.py describes each), unless set as an environment variable
  RUN_GROUPS  SEEDS  ALGORITHM  SKILL_SAMPLING  TEXT_BACKEND  BASE_SKILLS  EVAL_EPISODES  SAVE_INTERVAL
  LEARN_STEPS  LEARNABILITY_EMBEDDING  LEARN_EVAL_INTERVAL  TRAIN_EVAL_INTERVAL
  OPERATOR_EMBEDDINGS  OPERATOR_NAMES (exclude, include)  RQ1_STEPS (a number, or auto)  RQ1_BUDGET_MARGIN
  REPAIRS (FROM:TO pairs)  REPAIR_MODES  REPAIR_HIDDEN_EMBEDDINGS  REPAIR_STEPS  REPAIR_EVAL_INTERVAL
  RQ2_ORDERS (STAGE,STAGE,..., a stage's skills joined by +)  RQ2_SHUFFLES  RQ2_FIRST_STEPS  RQ2_STAGE_STEPS
  RQ2_EVAL_INTERVAL
  e.g. OPERATOR_EMBEDDINGS="one-hot text" SEEDS="0 1 2" REPAIRS="unstack:unstack-raised"
       RQ2_ORDERS="pickup+unstack,unstack-raised pickup,unstack" RQ2_SHUFFLES=0
Machine settings (environment variables, defaults in brackets)
  OUT [outputs/overnight_<date>]  JOBS [2] runs at once  DEVICE [auto]  THREADS [cores / JOBS] torch threads per run
  DRY_RUN [0]
"""

import argparse
import csv
import inspect
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from cam.experiments.representation_overlap import overlap as overlap_levels, report as overlap_report
from cam.experiments.research_experiments import EXPERIMENTS, literals, untrained_literals
from cam.skills.registry import SKILL_REGISTRY

ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").exists() else sys.executable


def command_line() -> argparse.Namespace:
    """The runner's arguments (imported as a module, e.g. by the tests: the defaults)."""
    parser = argparse.ArgumentParser(description="Run an experiment's plan of training runs (see this file's header).")
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="rq3-repair-minimal",
                        help="the experiment to run (default rq3-repair-minimal); --list describes each")
    parser.add_argument("--list", action="store_true", help="describe every experiment and its settings, run nothing")
    return parser.parse_args(sys.argv[1:] if __name__ == "__main__" else [])


ARGUMENTS = command_line()
EXPERIMENT = EXPERIMENTS[ARGUMENTS.experiment]
OVERRIDES: dict[str, str] = {}  # the experiment's settings set by environment variables instead


def setting_text(value) -> str:
    """An experiment's setting as its environment variable would give it: lists space-separated, pairs FROM:TO."""
    if isinstance(value, tuple):
        return " ".join(":".join(item) if isinstance(item, tuple) else str(item) for item in value)
    return str(value)


def setting(name: str, default: str | None = None) -> str:
    """The environment variable name, else the experiment's setting, else default (machine settings)."""
    experiment_settings = EXPERIMENT.settings()
    if name in os.environ:
        if name in experiment_settings:
            OVERRIDES[name] = os.environ[name]
        return os.environ[name]
    return setting_text(experiment_settings[name]) if name in experiment_settings else default


OUT = Path(setting("OUT", f"outputs/overnight_{date.today():%Y%m%d}"))
RUN_GROUPS = setting("RUN_GROUPS").split()
ALGORITHM = setting("ALGORITHM")
SKILL_SAMPLING = setting("SKILL_SAMPLING")
SEEDS = [int(seed) for seed in setting("SEEDS").split()]
OPERATOR_EMBEDDINGS = setting("OPERATOR_EMBEDDINGS").split()
OPERATOR_NAMES = setting("OPERATOR_NAMES").split()
REPAIR_HIDDEN_EMBEDDINGS = setting("REPAIR_HIDDEN_EMBEDDINGS").split()
TEXT_BACKEND = setting("TEXT_BACKEND")
BASE_SKILLS = setting("BASE_SKILLS").split()  # the training skills; held-out operators (REPAIRS' TO) only in repairs
REPAIRS = [tuple(pair.split(":")) for pair in setting("REPAIRS").split()]
REPAIRED = [to for _, to in REPAIRS]
REPAIR_MODES = setting("REPAIR_MODES").split()
REPAIR_EVAL_INTERVAL = int(setting("REPAIR_EVAL_INTERVAL"))
LEARN_EVAL_INTERVAL = int(setting("LEARN_EVAL_INTERVAL"))
LEARN_STEPS = int(setting("LEARN_STEPS"))
LEARNABILITY_EMBEDDING = setting("LEARNABILITY_EMBEDDING")
RQ1_AUTO = setting("RQ1_STEPS") == "auto"
RQ1_BUDGET_MARGIN = float(setting("RQ1_BUDGET_MARGIN"))
REPAIR_STEPS = int(setting("REPAIR_STEPS"))
RQ2_FIRST_STEPS = int(setting("RQ2_FIRST_STEPS"))
RQ2_STAGE_STEPS = int(setting("RQ2_STAGE_STEPS"))
RQ2_EVAL_INTERVAL = int(setting("RQ2_EVAL_INTERVAL"))
TRAIN_EVAL_INTERVAL = int(setting("TRAIN_EVAL_INTERVAL"))
EVAL_EPISODES = int(setting("EVAL_EPISODES"))
SAVE_INTERVAL = int(setting("SAVE_INTERVAL"))
DEVICE = setting("DEVICE", "auto")
JOBS = max(1, int(setting("JOBS", "2")))
THREADS = setting("THREADS", str(max(1, (os.cpu_count() or 1) // JOBS)))  # no CPU oversubscription
DRY_RUN = setting("DRY_RUN", "0") == "1"
STATUS_EVERY = 60 if sys.stdout.isatty() else 600  # seconds between progress reports while runs go

# Operator embedding conditions: name -> (main.py options, what the policy is told about the skill).
OPERATOR_EMBEDDING_OPTIONS = {
    "one-hot": (["--operator-encoder", "one-hot"], "a one-hot skill ID (no structure)"),
    "random": (["--operator-encoder", "random"], "a fixed random vector per skill name (no structure)"),
    "random-trainable": (["--operator-encoder", "random", "--trainable-operator-embedding"],
                         "a learnable vector per skill, starting random (no structure)"),
    "multi-hot": (["--operator-encoder", "multi-hot"], "the operator's literals as a multi-hot vector"),
    "multi-hot-trainable": (["--operator-encoder", "multi-hot", "--trainable-operator-embedding"],
                            "the operator's literals as a multi-hot vector, through a learnable layer"),
    "text": (["--operator-encoder", "text", "--operator-text", "paragraph"],
             f"the operator described in a paragraph, embedded by the {TEXT_BACKEND} text model"),
    "compositional-tree": (["--operator-encoder", "compositional", "--compositional-architecture", "tree"],
                           "the operator's structure, composed by learned functions following the PDDL syntax"),
    "compositional-slots": (["--operator-encoder", "compositional", "--compositional-architecture", "slots"],
                            "the operator's components as tokens, grouped by slot attention"),
    "compositional-geometric": (
        ["--operator-encoder", "compositional", "--compositional-architecture", "geometric"],
        "the operator's components on a grid, read by a CNN and slot attention",
    ),
}
# The embeddings that carry the operator's name, by --operator-encoder: the main.py options that leave it out, which
# OPERATOR_NAMES' exclude adds (the name can carry a repair by itself: "pickup-raised" shares "raised" with
# "unstack-raised").
NAME_EXCLUDED_OPTIONS = {"text": ["--operator-text-name", "none"], "compositional": ["--compositional-name", "none"]}
# Rough single-run SAC steps/s per condition on a CPU, for the first time estimate; replaced by the speeds
# measured as runs finish.
EXPECTED_RATE = {"compositional-tree": 24, "compositional-slots": 23, "compositional-geometric": 18}
DEFAULT_RATE = 40
COMMON = [
    "--algorithm", ALGORITHM, "--reward", "shaped", "--num-blocks", "2", "--device", DEVICE,
    "--skill-sampling", SKILL_SAMPLING,
    "--eval-interval", str(TRAIN_EVAL_INTERVAL), "--eval-episodes", str(EVAL_EPISODES),
    "--save-interval", str(SAVE_INTERVAL), "--text-backend", TEXT_BACKEND,
]


@dataclass
class Run:
    label: str
    group: str
    title: str
    what: str
    why: str
    steps: int  # training steps this run adds
    condition: str  # its operator embedding (with or without the name: the same speed), for the time estimate
    output: Path  # new runs: --output-dir (every rq2 stage: its stage 1's); repair runs: the run directory itself
    command: list[str]
    copy_from: Path | None = None  # repair runs: the rq1 output directory holding the run to continue
    depends_on: tuple[str, ...] = ()  # labels of runs that must end first
    added: tuple[str, ...] = ()  # the skills added to the run continued: a repair's TO, a later rq2 stage's skills
    stage: int = 0  # rq2 runs: the stage's number in its order (1: the new run, which the later stages continue)
    embedding_options: tuple[str, ...] = ()  # main.py options of its operator embedding (a repair's: its rq1 run's)

    @property
    def log(self) -> Path:
        return OUT / "logs" / f"{self.label}.log"

    def finished(self) -> bool:
        return self.log.exists() and "training finished" in self.log.read_text(errors="replace")

    def stop_reason(self) -> str | None:
        """Why the run stopped before its step budget, as it last logged it (MetricsCallback); None if it did not."""
        log = self.log.read_text(errors="replace") if self.log.exists() else ""
        reasons = re.findall(r"stopping training at step \d+: (.*)", log)
        return reasons[-1] if reasons else None

    def run_directory(self) -> Path | None:
        if self.copy_from is not None:
            return self.output if (self.output / "config.json").exists() else None
        found = sorted(path.parent for path in self.output.glob("*/config.json"))
        return found[0] if found else None


def condition_options(condition: str) -> list[str]:
    if condition not in OPERATOR_EMBEDDING_OPTIONS:
        sys.exit(f"unknown condition {condition!r}; known: {', '.join(OPERATOR_EMBEDDING_OPTIONS)}"
                 + ("; the -no-name runs come from OPERATOR_NAMES=exclude" if condition.endswith("-no-name") else ""))
    return OPERATOR_EMBEDDING_OPTIONS[condition][0]


def name_excluded_options(condition: str) -> list[str] | None:
    """The main.py options that leave the operator's name out of the condition's embedding; None if it has none."""
    options = condition_options(condition)
    return NAME_EXCLUDED_OPTIONS.get(options[options.index("--operator-encoder") + 1])


def conditions() -> list[tuple[str, str, list[str], str]]:
    """The rq1 and repair runs' conditions: (label, operator embedding, main.py options, what the policy is told about
    the skill). An embedding with the operator's name runs once per OPERATOR_NAMES (exclude: labelled
    <embedding>-no-name; include: <embedding>), any other once."""
    planned = []
    for embedding in OPERATOR_EMBEDDINGS:
        options, description = condition_options(embedding), OPERATOR_EMBEDDING_OPTIONS[embedding][1]
        excluded = name_excluded_options(embedding)
        if excluded is None:
            planned.append((embedding, embedding, options, description))
            continue
        for choice in OPERATOR_NAMES:
            planned.append((f"{embedding}-no-name", embedding, options + excluded,
                            f"{description} (without the operator's name)") if choice == "exclude" else
                           (embedding, embedding, options, f"{description} (with the operator's name)"))
    return planned


def repair_difference(repair_from: str, repair_to: str) -> str:
    """What repair_to changes relative to repair_from, e.g. 'adds effect (raised ?o)'."""
    before, after = literals(repair_from), literals(repair_to)
    changes = [f"adds {literal}" for literal in sorted(after - before)]
    changes += [f"removes {literal}" for literal in sorted(before - after)]
    return ", ".join(changes) or "no literal changes"


def rq1_budget(steps_to_threshold: dict[str, int | None], margin: float, learn_steps: int, interval: int) -> int:
    """RQ1_STEPS=auto: margin x the sum of the steps each skill needed alone to reach the threshold (at least one
    evaluation interval, a learnability run's resolution; learn_steps, all it had, if it never did), rounded up to
    whole evaluation intervals."""
    needed = sum(learn_steps if steps is None else max(steps, interval) for steps in steps_to_threshold.values())
    return math.ceil(margin * needed / interval) * interval


def learnability_steps(skills: list[str]) -> tuple[dict[str, int | None], list[str]]:
    """Each skill's deterministic steps to threshold in its learnability run under OUT (None: never reached),
    and the skills without a learnability result (also None)."""
    steps, missing = {}, []
    for skill in skills:
        found = sorted((OUT / "learnability" / skill).glob("*/metrics.json"))
        if not found:
            steps[skill] = None
            missing.append(skill)
            continue
        steps[skill] = json.loads(found[0].read_text())["summary"]["deterministic"].get(skill, {}).get("steps_to_threshold")
    return steps, missing


# rq1 steps: the setting, or with RQ1_STEPS=auto the most the budget can be until the learnability runs are done
RQ1_STEPS = (rq1_budget(dict.fromkeys(BASE_SKILLS), RQ1_BUDGET_MARGIN, LEARN_STEPS, LEARN_EVAL_INTERVAL) if RQ1_AUTO
             else int(setting("RQ1_STEPS")))


def order_stages(order: str) -> list[list[str]]:
    """An order as RQ2_ORDERS writes it, e.g. "pickup+unstack,unstack-raised", as its stages, each a list of skills:
    [[pickup, unstack], [unstack-raised]]."""
    return [stage.split("+") for stage in order.split(",")]


def order_name(order: list[list[str]]) -> str:
    """The order as RQ2_ORDERS writes it: its name in directories and labels."""
    return ",".join("+".join(stage) for stage in order)


def shuffled_orders(skills: list[str], count: int, seed: int = 0) -> list[list[list[str]]]:
    """count different orders of the skills, one skill per stage (at most as many as there are), shuffled from a
    fixed seed: the same orders on every invocation, and a larger count keeps a smaller one's orders first."""
    generator, orders = random.Random(seed), []
    while len(orders) < min(count, math.factorial(len(skills))):
        order = [[skill] for skill in generator.sample(skills, len(skills))]
        if order not in orders:
            orders.append(order)
    return orders


def stage_coverage(order: list[list[str]], number: int) -> dict[str, float]:
    """Each skill the order's stage `number` (from 1) adds, with the fraction of its literals the stages before it have
    (representation_overlap.py's coverage at the literal level): zero-shot success is only possible at 1."""
    before = [skill for stage in order[:number - 1] for skill in stage]
    literals_level = next(level for level in overlap_levels(before, order[number - 1]) if level.level == "literals")
    return literals_level.coverage


def added_at(metrics: dict, skills) -> int | None:
    """The step at which a run (its metrics.json) first evaluated any of the skills: where they were added, their
    zero-shot point; None if it has not."""
    points = metrics["curves"]["deterministic"]["points"]
    return next((point["step"] for point in points if set(skills) & set(point["result"])), None)


RQ2_ORDERS = ([order_stages(order) for order in setting("RQ2_ORDERS").split()]
              + shuffled_orders(BASE_SKILLS, int(setting("RQ2_SHUFFLES"))))


def link_replay_buffer(source: str, destination: str) -> str:
    """copytree's copy function for a repair: the rq1 run's replay buffer (about 1 GB, only read) is hard-linked
    instead of copied, everything else copied. Saving writes a new file and renames it, so the link is replaced,
    never written through."""
    if Path(source).name == "replay_buffer.pkl":
        try:
            os.link(source, destination)
            return destination
        except OSError:
            pass
    return shutil.copy2(source, destination)


def build_plan() -> list[Run]:
    runs: list[Run] = []
    repairs: list[Run] = []  # after every rq1 run, so each condition's rq1 result comes before any repair
    skill_list = ", ".join(BASE_SKILLS)
    if "learnability" in RUN_GROUPS:
        for skill in dict.fromkeys(BASE_SKILLS + REPAIRED):
            output = OUT / "learnability" / skill
            runs.append(Run(
                label=f"learnability_{skill}", group="learnability", title=f"Learnability · {skill}",
                what=(f"one {ALGORITHM.upper()} policy trained on {skill} alone ({LEARNABILITY_EMBEDDING} embedding), "
                      f"{LEARN_STEPS:,} steps"),
                why=(f"can {skill} be learned at all, and how fast? If not, no result that uses it means anything. "
                     "Also the single-skill baseline for RQ4 (hypernetworks vs individual policies)."),
                steps=LEARN_STEPS, condition=LEARNABILITY_EMBEDDING, output=output,
                embedding_options=tuple(condition_options(LEARNABILITY_EMBEDDING)),
                command=["main.py", "--skills", skill, *condition_options(LEARNABILITY_EMBEDDING),
                         "--total-timesteps", str(LEARN_STEPS), "--seed", "0", *COMMON, "--output-dir", str(output),
                         "--eval-interval", str(LEARN_EVAL_INTERVAL)],
            ))
    for seed in SEEDS:
        for condition, embedding, options, description in conditions():
            rq1_label, rq1_output = f"rq1_{condition}_seed{seed}", OUT / "rq1" / f"{condition}_seed{seed}"
            if "rq1" in RUN_GROUPS:
                runs.append(Run(
                    label=rq1_label, group="rq1", title=f"RQ1 · {condition} · seed {seed}",
                    what=(f"one shared {ALGORITHM.upper()} policy on {len(BASE_SKILLS)} skills ({skill_list}), "
                          + ("a budget set from the learnability runs (RQ1_STEPS=auto)" if RQ1_AUTO else
                             f"{RQ1_STEPS:,} steps")
                          + f"; the policy is told which skill to do through {description}"),
                    why=("RQ1: does conditioning on this representation of the operator make multi-skill learning "
                         "faster or better than an arbitrary skill ID? Compared across conditions on steps to "
                         "threshold, AUC and final success."),
                    steps=RQ1_STEPS, condition=embedding, output=rq1_output, embedding_options=tuple(options),
                    depends_on=tuple(f"learnability_{skill}" for skill in BASE_SKILLS)
                    if RQ1_AUTO and "learnability" in RUN_GROUPS else (),
                    command=["main.py", "--skills", *BASE_SKILLS, *options, "--total-timesteps", str(RQ1_STEPS),
                             "--seed", str(seed), *COMMON, "--output-dir", str(rq1_output),
                             "--save-replay-buffer"],  # the repairs continue from it
                ))
            if "repair" not in RUN_GROUPS:
                continue
            for (repair_from, repair_to), mode in ((pair, mode) for pair in REPAIRS for mode in REPAIR_MODES):
                difference = repair_difference(repair_from, repair_to)
                skills = [repair_to if skill == repair_from else skill for skill in BASE_SKILLS]
                variants = ["repair-visible"] + (["repair-hidden"] if embedding in REPAIR_HIDDEN_EMBEDDINGS else [])
                trains = (f"trains only {repair_to} until it reaches the success threshold (the other skills are "
                          "evaluated, not trained)" if mode == "isolated" else
                          f"trains every skill, as the RQ1 run did, until {repair_to} reaches the success threshold "
                          "and every other skill is back at its level before the repair")
                for variant in variants:
                    run_directory = OUT / "repair" / repair_to / f"{condition}_{variant}_{mode}_seed{seed}"
                    hidden = variant == "repair-hidden"
                    repairs.append(Run(
                        label=f"repair_{repair_to}_{condition}_{variant}_{mode}_seed{seed}", group="repair",
                        title=(f"Repair (RQ3) · {repair_from} -> {repair_to} · {condition} · {variant} · {mode} · "
                               f"seed {seed}"),
                        what=(f"continues the RQ1 {condition} policy with {repair_from} replaced by {repair_to} "
                              f"({difference}): evaluates {repair_to} zero-shot, then {trains} (at most "
                              f"{REPAIR_STEPS:,} steps)"),
                        why=(f"control: {repair_to} gets exactly {repair_from}'s embedding, so the policy cannot see "
                             "the repair. If this succeeds zero-shot, the repair needs no new behaviour and the "
                             "repair test is invalid." if hidden else
                             f"RQ3: does {description} let the policy carry out the repaired operator zero-shot, "
                             "or learn it faster than the other conditions?"),
                        steps=REPAIR_STEPS, condition=embedding, output=run_directory, copy_from=rq1_output,
                        depends_on=(rq1_label,) if "rq1" in RUN_GROUPS else (), added=(repair_to,),
                        embedding_options=tuple(options),
                        command=["main.py", "--run-directory", str(run_directory), "--skills", *skills,
                                 "--stop-skills", repair_to, "--repair-mode", mode,
                                 "--total-timesteps", str(REPAIR_STEPS), "--eval-interval", str(REPAIR_EVAL_INTERVAL),
                                 "--device", DEVICE, "--no-save-replay-buffer",  # the rq1 run's, linked
                                 *(["--operator-identity-aliases", f"{repair_to}={repair_from}"] if hidden else [])],
                    ))
    rq2 = [run for seed in SEEDS for order in RQ2_ORDERS for planned in conditions()
           for run in rq2_chain(order, *planned, seed)] if "rq2" in RUN_GROUPS else []
    return runs + repairs + rq2


def rq2_chain(order: list[list[str]], condition: str, embedding: str, options: list[str], description: str,
              seed: int) -> list[Run]:
    """An rq2 chain: stage 1 a new run on its skills, and each later stage, after the one before, that run continued
    with its skills added (Runner.start lets a stage begin only if the one before stopped at its threshold).

    The run is continued in place, not copied per stage as repairs copy their rq1 run: several repairs branch from one
    rq1 run, while a chain never branches, so a copy would only keep each stage's last model, at the cost of a replay
    buffer (about 1 GB) per stage. In place, one metrics.json holds the order's whole learning curve and every
    addition (each earlier one's retention in earlier_retention), and one TensorBoard log its training."""
    name = order_name(order)
    # the order as the run's parent directory: review_runs.py names a condition by a run's parent and its name
    # without the seed, so a condition's seeds go together and orders apart
    output = OUT / "rq2" / f"{condition}_seed{seed}" / name
    label = f"rq2_{name}_{condition}_seed{seed}"
    runs = []
    for number, stage in enumerate(order, 1):
        before = [skill for earlier in order[:number - 1] for skill in earlier]
        title = (f"RQ2 · {condition} · seed {seed} · stage {number}: {'+ ' if before else ''}{' + '.join(stage)}"
                 f" · order {name}")
        if number == 1:
            runs.append(Run(
                label=f"{label}_stage1", group="rq2", title=title,
                what=(f"a new shared {ALGORITHM.upper()} policy on {', '.join(stage)}, until each reaches the success "
                      f"threshold (at most {RQ2_FIRST_STEPS:,} steps); the policy is told which skill to do through "
                      f"{description}"),
                why=f"RQ2: the policy that the order's {len(order) - 1} later stages add their skills to, in turn",
                steps=RQ2_FIRST_STEPS, condition=embedding, output=output, stage=1,
                embedding_options=tuple(options),
                command=["main.py", "--skills", *stage, *options, "--total-timesteps", str(RQ2_FIRST_STEPS),
                         "--seed", str(seed), *COMMON, "--output-dir", str(output),
                         "--eval-interval", str(RQ2_EVAL_INTERVAL), "--stop-skills", *stage,
                         "--save-replay-buffer"],  # the later stages keep replaying the earlier skills
            ))
            continue
        coverage = ", ".join(f"{skill} {value:.2f}" for skill, value in stage_coverage(order, number).items())
        runs.append(Run(
            label=f"{label}_stage{number}", group="rq2", title=title,
            what=(f"continues the stage {number - 1} policy with {', '.join(stage)} added (literal coverage by the "
                  f"skills before: {coverage}): evaluates it zero-shot, then trains every skill until each added one "
                  f"reaches the success threshold and every other skill is back at its level before (at most "
                  f"{RQ2_STAGE_STEPS:,} steps)"),
            why=(f"RQ2: does {description} let the policy carry out {', '.join(stage)} zero-shot (possible only at "
                 "coverage 1), or learn it in fewer steps than the other conditions?"),
            steps=RQ2_STAGE_STEPS, condition=embedding, output=output, depends_on=(f"{label}_stage{number - 1}",),
            added=tuple(stage), stage=number, embedding_options=tuple(options),
            # the directory holding stage 1's run: Runner.prepare puts the run's own in its place
            command=["main.py", "--run-directory", str(output), "--skills", *before, *stage, "--stop-skills", *stage,
                     "--repair-mode", "retain", "--total-timesteps", str(RQ2_STAGE_STEPS),
                     "--eval-interval", str(RQ2_EVAL_INTERVAL), "--device", DEVICE],
        ))
    return runs


def duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, minutes = divmod(seconds // 60, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds % 60:02d}s"


def say(text: str = "") -> None:
    print(text, flush=True)


def fail(message: str) -> None:
    say(f"\n✗ {message}")
    sys.exit(1)


# ---------------------------------------------------------------- checks before anything long starts


def preflight(runs: list[Run]) -> None:
    say("Checks before starting")
    import torch

    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        fail(f"DEVICE={DEVICE} but torch has no CUDA (torch {torch.__version__}); use DEVICE=auto or cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    device = DEVICE if DEVICE != "auto" else ("cuda" if gpu else "cpu")
    say(f"  ✓ device: {device}" + (f" ({gpu})" if gpu else f" (torch {torch.__version__}; no CUDA GPU)"))

    uses_repair_skill = any(run.group in ("learnability", "repair") for run in runs)
    rq2_skills = {skill for order in RQ2_ORDERS for stage in order for skill in stage} if "rq2" in RUN_GROUPS else set()
    skills = sorted(set(BASE_SKILLS) | (set(REPAIRED) if uses_repair_skill else set()) | rq2_skills)
    check_text_backend(skills, {run.embedding_options for run in runs})
    check_scripted_policies(skills)
    free = shutil.disk_usage(OUT if OUT.exists() else ROOT).free / 1e9
    say(f"  ✓ free disk space: {free:.0f} GB" + ("  (low: runs save models and logs)" if free < 10 else ""))


def check_repair_modes() -> None:
    unknown = [mode for mode in REPAIR_MODES if mode not in ("isolated", "retain")]
    if unknown:
        sys.exit(f"unknown REPAIR_MODES {unknown}; known: isolated, retain")


def check_operator_names() -> None:
    unknown = [choice for choice in OPERATOR_NAMES if choice not in ("exclude", "include")]
    if unknown or not OPERATOR_NAMES:
        sys.exit(f"unknown OPERATOR_NAMES {unknown or OPERATOR_NAMES}; known: exclude, include (or both)")


def check_repairs() -> None:
    """Each repair replaces a base skill with an operator held out of training, every literal of which a base skill
    has: otherwise the repair asks for something the policy never had the chance to learn."""
    if "repair" not in RUN_GROUPS:
        return
    problems = []
    for repair_from, repair_to in REPAIRS:
        if repair_from not in BASE_SKILLS:
            problems.append(f"{repair_from} -> {repair_to}: {repair_from} is not in BASE_SKILLS")
        if repair_to in BASE_SKILLS:
            problems.append(f"{repair_from} -> {repair_to}: {repair_to} is in BASE_SKILLS, so it is not held out")
        for literal in untrained_literals(repair_to, BASE_SKILLS):
            carriers = [skill for skill in SKILL_REGISTRY if skill != repair_to and literal in literals(skill)]
            problems.append(f"{repair_from} -> {repair_to}: no base skill has {literal}"
                            + (f" (add one of: {', '.join(carriers)})" if carriers else " (no skill has it)"))
    if problems:
        sys.exit("repairs that test nothing learnable:\n  " + "\n  ".join(problems))


def check_orders() -> None:
    """Each rq2 order names registered skills, each in one stage. Unlike a repair, an added skill may have literals no
    stage before it trained: how much of it they cover varies with the order by design, and the plan shows it."""
    if "rq2" not in RUN_GROUPS:
        return
    problems = [] if RQ2_ORDERS else ["no orders: set RQ2_ORDERS or RQ2_SHUFFLES"]
    for order in RQ2_ORDERS:
        skills = [skill for stage in order for skill in stage]
        unknown = [skill for skill in skills if skill not in SKILL_REGISTRY]
        repeated = sorted({skill for skill in skills if skills.count(skill) > 1})
        if unknown:
            problems.append(f"{order_name(order)}: unknown skills {unknown}")
        if repeated:
            problems.append(f"{order_name(order)}: {', '.join(repeated)} in more than one stage")
    if problems:
        sys.exit("rq2 orders that cannot run:\n  " + "\n  ".join(problems))


def check_text_backend(skills: list[str], embeddings: set[tuple[str, ...]]) -> None:
    """Embed every operator text the runs need now, through the same cache the runs use, so a missing key
    or network fails here and training never waits on the API: the texts that each of the runs' operator
    embeddings (main.py options) embeds, so with or without the operator's name as the runs have it."""
    from cam.experiments.environment_setup import OPERATOR_ENCODERS, text_backend
    from cam.skills.registry import build_skill
    from main import parse_args

    configs = [parse_args(["--skills", *skills, "--text-backend", TEXT_BACKEND, *options])
               for options in sorted(embeddings)]
    texts = [config for config in configs if config["operator_encoder"] == "text"]
    names = [config for config in configs
             if config["operator_encoder"] == "compositional" and config["compositional_name"] == "text"]
    if not texts and not names:
        return  # no run embeds a text
    try:
        operators = [build_skill(skill, configs[0]) for skill in skills]
        for config in texts:  # the operator's text (text conditions)
            encoder = OPERATOR_ENCODERS["text"](config, operators)
            for skill in operators:
                encoder.encode(skill.symbolic_action_model)
        if names:  # the name embedding (compositional conditions), the same for every architecture
            backend = text_backend(names[0], names[0]["component_embedding_dim"])
            for skill in operators:
                backend.embed_text(skill.name)
    except Exception as error:  # report any failure (no key, no network, no openai package) and stop
        fail(f"text backend {TEXT_BACKEND!r} is not usable: {type(error).__name__}: {error}\n"
             "  set OPENAI_API_KEY or put the key in ~/.secrets/openai_api_key, or run with TEXT_BACKEND=mock")
    embedded = [f"{len(operators)} operator {config['operator_text']}s "
                f"({'without' if config['operator_text_name'] == 'none' else 'with'} the name)" for config in texts]
    embedded += [f"{len(operators)} operator names"] if names else []
    say(f"  ✓ text backend {TEXT_BACKEND}: {' and '.join(embedded)} embedded and cached "
        f"({configs[0]['text_embedding_cache']})")


def check_scripted_policies(skills: list[str]) -> None:
    """Every skill's scripted policy must succeed at least once in 3 episodes (2 blocks) after its setup chain:
    proof that the skill's setup and predicates work. Not every episode: a straight scripted push can knock the
    other block aside, which is a failure."""
    failed = []
    for skill in skills:
        result = subprocess.run(
            # 150 steps: the scripted push moves slowly on purpose (gentle contact), so diagonals need >100
            [PYTHON, "scripts/run_skill.py", "--skill", skill, "--num-blocks", "2", "--episodes", "3", "--max-steps", "150"],
            cwd=ROOT, capture_output=True, text=True,
        )
        if not re.search(r"success rate: [1-3]/3", result.stdout):
            failed.append(f"{skill}: {(result.stdout + result.stderr).strip().splitlines()[-1:]}")
    if failed:
        fail("scripted policies failed (the skill's setup or predicates are broken):\n  " + "\n  ".join(failed))
    say(f"  ✓ scripted policies succeed (at least 1 of 3 episodes) for all {len(skills)} skills: {', '.join(skills)}")


# ---------------------------------------------------------------- running


@dataclass
class Job:
    """A run in progress: its child process and what is needed to report its progress."""

    run: Run
    index: int
    process: subprocess.Popen
    log: object
    started: float
    samples: list  # recent (time, steps done) pairs, for the speed


class Runner:
    """Executes the plan: up to JOBS runs at once, each announced, followed and reported."""

    def __init__(self, runs: list[Run]):
        self.runs = runs
        self.rates: dict[str, float] = {}  # measured steps/s per condition (at the current parallelism)
        self.done: set[str] = set()
        self.failed: set[str] = set()
        self.ended: set[str] = set()  # of the failed: rq2 stages not run because their chain ended before them
        self.started = time.monotonic()
        self.jobs: list[Job] = []
        self.last_status = 0.0
        self.rq1_steps: int | None = None  # RQ1_STEPS=auto: the budget, once the learnability runs are done

    # ---- time estimates

    def rate(self, condition: str) -> float:
        """Steps/s of one run of this condition: measured if one finished, else the rough default, scaled by
        how fast this machine was on the conditions measured so far."""
        if condition in self.rates:
            return self.rates[condition]
        if self.rates:
            measured = [self.rates[c] / EXPECTED_RATE.get(c, DEFAULT_RATE) for c in self.rates]
            return EXPECTED_RATE.get(condition, DEFAULT_RATE) * sum(measured) / len(measured)
        return EXPECTED_RATE.get(condition, DEFAULT_RATE)

    def time_left(self) -> float:
        """Seconds until the plan is done: the runs not finished, spread over JOBS parallel slots."""
        seconds = 0.0
        for run in self.runs:
            if run.label in self.done or run.label in self.failed:
                continue
            job = next((job for job in self.jobs if job.run is run), None)
            left = run.steps - (self.progress(run)[0] or 0 if job else 0)
            seconds += left / self.rate(run.condition)
        return seconds / JOBS

    def print_plan(self) -> None:
        say(f"\nPlan: {len(self.runs)} runs, {JOBS} at a time ({ALGORITHM}, device {DEVICE}, "
            f"{THREADS} torch threads each), output in {OUT}")
        for index, run in enumerate(self.runs, 1):
            state = "done" if run.finished() else f"{run.steps // 1000}k steps"
            if run.group == "rq1" and RQ1_AUTO and self.rq1_steps is None and not run.finished():
                state = f"auto (at most {run.steps // 1000}k steps)"
            after = (f"  (after {', '.join(run.depends_on)})" if len(run.depends_on) <= 2 else
                     f"  (after {len(run.depends_on)} runs: {run.depends_on[0]}, ...)") if run.depends_on else ""
            say(f"  {index:2d}. {run.title:66s} {state}{after}")
        say(f"Estimated time: ~{duration(self.time_left())} (a first guess from typical CPU speeds; refined from "
            "the measured speed as runs finish)")

    # ---- the main loop

    def execute(self) -> None:
        pending = list(self.runs)
        while pending or self.jobs:
            for run in list(pending):  # start every run that can start, in plan order
                if len(self.jobs) >= JOBS:
                    break
                if any(label not in self.done | self.failed for label in run.depends_on):
                    continue  # waits for the runs it needs
                pending.remove(run)
                self.start(run, self.runs.index(run) + 1)
            if not self.jobs:
                if pending:  # only runs whose dependency is missing from the plan: cannot happen, but never spin
                    for run in pending:
                        say(f"\n✗ {run.title}: depends on {run.depends_on}, which is not in the plan; skipped")
                    break
                continue
            time.sleep(5)
            for job in list(self.jobs):
                done, _ = self.progress(job.run)
                if done is not None:
                    job.samples = (job.samples + [(time.monotonic(), done)])[-24:]  # the last ~2 minutes
                if job.process.poll() is not None:
                    self.jobs.remove(job)
                    self.finish(job)
            self.show_status()

    def start(self, run: Run, index: int) -> None:
        """Announce the run (what and why), then launch it, unless it is done or cannot run."""
        say(f"\n━━ [{index}/{len(self.runs)}] {run.title}")
        say(f"   what: {run.what}")
        say(f"   why:  {run.why}")
        say(f"   log:  {run.log}")
        if run.finished():
            say("   ✓ already finished (skipped)")
            self.done.add(run.label)
            self.report(run)
            return
        failed_dependencies = [label for label in run.depends_on if label in self.failed]
        if (run.copy_from is not None or run.stage > 1) and failed_dependencies:  # it continues their model
            if set(failed_dependencies) & self.ended:
                say("   ⤼ skipped: its chain ended before it")
                self.ended.add(run.label)
            else:
                say(f"   ⤼ skipped: {', '.join(failed_dependencies)} failed")
            self.failed.add(run.label)
            return
        if run.stage > 1 and self.chain_ends(run):
            self.failed.add(run.label)
            self.ended.add(run.label)
            return
        command = self.prepare(run)
        if command is None:
            self.done.add(run.label)
            return
        run.log.parent.mkdir(parents=True, exist_ok=True)
        log = open(run.log, "a")
        log.write(f"\n# {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(command)}\n")
        log.flush()
        environment = os.environ | {"OMP_NUM_THREADS": THREADS, "MKL_NUM_THREADS": THREADS}
        process = subprocess.Popen([PYTHON, *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        self.jobs.append(Job(run, index, process, log, time.monotonic(), []))
        say(f"   ▶ started ({len(self.jobs)} of {JOBS} slots busy)")

    def chain_ends(self, run: Run) -> bool:
        """A later rq2 stage starts only if the stage before stopped at its success threshold. Otherwise its chain ends
        here: say why, and record it in the run's chain_ended.txt for review_runs.py."""
        before = next(other for other in self.runs if other.label in run.depends_on)
        stopped = before.stop_reason()
        directory = run.run_directory()
        record = directory / "chain_ended.txt" if directory is not None else None
        if stopped and "reached success rate" in stopped:  # MetricsCallback's stop at the threshold
            if record is not None:
                record.unlink(missing_ok=True)
            return False
        ended = (f"chain ended: stage {before.stage} did not reach the success threshold ("
                 + (f"it stopped: {stopped}" if stopped else f"its {before.steps:,} steps ran out")
                 + f"), so stage {run.stage} and any after it did not run")
        say(f"   ⤼ {ended}")
        if record is not None:
            record.write_text(ended + "\n")
        return True

    def set_rq1_budget(self, run: Run) -> None:
        """RQ1_STEPS=auto: the rq1 run's budget from the learnability results, the same for every rq1 run."""
        if self.rq1_steps is None:
            steps, missing = learnability_steps(BASE_SKILLS)
            self.rq1_steps = rq1_budget(steps, RQ1_BUDGET_MARGIN, LEARN_STEPS, LEARN_EVAL_INTERVAL)
            needed = ", ".join(f"{skill} {'-' if value is None else f'{value // 1000}k'}" for skill, value in steps.items())
            say(f"   budget (RQ1_STEPS=auto): {self.rq1_steps:,} steps = {RQ1_BUDGET_MARGIN} x the steps each skill "
                f"needed alone ({needed}; '-' counts as {LEARN_STEPS // 1000}k)")
            if missing:
                say(f"   ! no learnability result for {', '.join(missing)} in {OUT / 'learnability'}; counted as "
                    f"{LEARN_STEPS // 1000}k")
        run.steps = self.rq1_steps
        run.command[run.command.index("--total-timesteps") + 1] = str(self.rq1_steps)

    def prepare(self, run: Run) -> list[str] | None:
        """The command to run: a new run, a repair copied from its rq1 run (also when interrupted), a later rq2 stage
        continuing the run of its chain, or a resume of an interrupted run."""
        if run.group == "rq1" and RQ1_AUTO:
            self.set_rq1_budget(run)
        run_directory = run.run_directory()
        command = list(run.command)
        if run.stage > 1:  # continues the run its stage 1 made, in place
            if run_directory is None:
                say(f"   ✗ no run in {run.output} to continue; skipped")
                self.failed.add(run.label)
                return None
            command[command.index("--run-directory") + 1] = str(run_directory)
        if run.copy_from is not None and run_directory is not None:
            # an interrupted repair starts over from its rq1 run: resumed, it would reload the rq1 run's replay buffer
            # (a repair saves none) with the replaced skill's transitions back in, and its config.json would no longer
            # show that it replaced a skill (main.py --reset-replaced-skills)
            say("   restarting an interrupted repair from its rq1 run")
            shutil.rmtree(run_directory)
            run_directory = None
        if run.copy_from is not None and run_directory is None:
            sources = sorted(path.parent for path in run.copy_from.glob("*/model.zip"))
            if not sources:
                say(f"   ✗ no finished rq1 run in {run.copy_from} to continue; skipped")
                self.failed.add(run.label)
                return None
            shutil.copytree(sources[0], run.output, copy_function=link_replay_buffer)
            return run.command
        if run_directory is not None and (run_directory / "model.zip").exists():  # interrupted earlier
            metrics = json.loads((run_directory / "metrics.json").read_text())
            done = metrics["progress"]["steps"]
            if run.stage > 1:  # a later rq2 stage: from where its skills were added (None: it has not started)
                base = added_at(metrics, run.added)
                if base is None:
                    return command
            else:
                base = 0
            left = run.steps - (done - base)
            if left <= 0:
                say("   ✓ already at its step budget")
                return None
            say(f"   resuming an interrupted run at step {done:,}: {left:,} steps left")
            if run.copy_from is None and run.stage <= 1:  # new run -> resume it in place
                command = ["main.py", "--run-directory", str(run_directory),
                           *run.command[run.command.index("--skills"):]]
                command = [part for part in command if part != "--output-dir" and part != str(run.output)]
            command[command.index("--total-timesteps") + 1] = str(left)
            return command
        if run_directory is not None:  # started but never saved: start over
            shutil.rmtree(run_directory)
        return run.command

    def finish(self, job: Job) -> None:
        """Report a run that ended: its outcome, or why it failed."""
        job.log.close()
        run, elapsed = job.run, time.monotonic() - job.started
        done, _ = self.progress(run)
        if done and elapsed > 60:
            self.rates[run.condition] = done / elapsed
        say(f"\n■ [{job.index}/{len(self.runs)}] {run.title}")
        if job.process.returncode != 0 or not run.finished():
            self.failed.add(run.label)
            with open(OUT / "failed.txt", "a") as failed:
                failed.write(run.label + "\n")
            say(f"   ✗ failed (exit code {job.process.returncode}) after {duration(elapsed)}; last lines of its log:")
            tail = run.log.read_text(errors="replace").replace("\r", "\n").strip().splitlines()[-12:]
            say("\n".join(f"     {line}" for line in tail))
        else:
            self.done.add(run.label)
            say(f"   ✓ finished in {duration(elapsed)}")
            self.report(run)
        finished = len(self.done) + len(self.failed)
        say(f"   overall: {finished}/{len(self.runs)} runs ended · elapsed {duration(time.monotonic() - self.started)}"
            f" · ~{duration(self.time_left())} left")

    def show_status(self) -> None:
        """Every STATUS_EVERY seconds: each running run's progress, and the time left overall."""
        now = time.monotonic()
        if not self.jobs or now - self.last_status < STATUS_EVERY:
            return
        self.last_status = now
        say(f"\n   ── {datetime.now():%H:%M} · {len(self.jobs)} running · "
            f"{len(self.done) + len(self.failed)}/{len(self.runs)} ended · ~{duration(self.time_left())} left overall")
        for job in self.jobs:
            done, total = self.progress(job.run)
            first, last = (job.samples[0], job.samples[-1]) if job.samples else (None, None)
            rate = (last[1] - first[1]) / (last[0] - first[0]) if first and last[0] > first[0] else 0.0
            left = f" · ~{duration((total - done) / rate)} left" if rate > 0 and done is not None else ""
            steps = f"{done:,}/{total:,} steps" if done is not None else "starting"
            say(f"      [{job.index:2d}] {job.run.title[:60]:60s} {steps} · {rate:.0f} steps/s{left}")

    # ---- reading a run's state

    @staticmethod
    def progress(run: Run) -> tuple[int | None, int]:
        """(steps done in this invocation, its total) from progress.json in the run's directory, which training writes
        every 1000 steps (ProgressFileCallback); (None, the run's steps) until it exists. The progress bar cannot
        serve: drawn only on a terminal, it appears in a run's log once, at the end."""
        directory = run.run_directory()
        try:
            progress = json.loads((directory / "progress.json").read_text())
            return int(progress["done"]), int(progress["total"])
        except (OSError, TypeError, ValueError, KeyError):
            return None, run.steps

    @staticmethod
    def report(run: Run) -> None:
        """The run's outcome from its metrics.json: per skill, final success and steps to threshold (and how many of
        those steps were spent on the skill itself); for a repair or a later rq2 stage, the added skills' and what the
        others forgot."""
        run_directory = run.run_directory()
        if run_directory is None or not (run_directory / "metrics.json").exists():
            return
        metrics = json.loads((run_directory / "metrics.json").read_text())
        summary = metrics["summary"]["deterministic"]
        if run.added:
            cells = [f"{skill} zero-shot {summary.get(skill, {}).get('zero_shot_success_rate')}, steps to threshold "
                     f"{summary.get(skill, {}).get('steps_to_threshold')} "
                     f"({summary.get(skill, {}).get('skill_steps_to_threshold')} on it)" for skill in run.added]
            stopped = run.stop_reason()
            say(f"   result: {'; '.join(cells)}" + (f"; stopped: {stopped}" if stopped else ""))
            # this addition's: a chain's run keeps its earlier additions' in earlier_retention
            step = added_at(metrics, run.added)
            retention = next((entry for entry in [*metrics.get("earlier_retention", []), metrics.get("retention")]
                              if entry and entry["added_at"] == step), {})
            others = retention.get("skills") or {}
            if others:
                mean = lambda key: sum(entry[key] or 0 for entry in others.values()) / len(others)  # noqa: E731
                back = retention.get("all_recovered_at")
                say(f"   other skills (mean): {mean('level_before'):.2f} before -> lowest {mean('lowest_after'):.2f} "
                    f"-> {mean('final'):.2f} at the end; "
                    + (f"all back after {back - retention['added_at']:,} steps" if back is not None else "not all back"))
            return
        def reached(values: dict) -> str:
            steps, on_skill = values.get("steps_to_threshold"), values.get("skill_steps_to_threshold")
            if steps is None:
                return ""
            return f" (95% at {steps // 1000}k" + (f", {on_skill // 1000}k on it" if on_skill is not None else "") + ")"

        cells = [f"{skill} {values.get('final_success_rate')}{reached(values)}" for skill, values in summary.items()]
        say(f"   result (final success, deterministic): {'; '.join(cells)}")

    def interrupt(self, *_) -> None:
        """Ctrl+C: stop every running run (each saves first), then exit; the same command resumes."""
        say(f"\n   stopping {len(self.jobs)} running run(s); each saves first ...")
        for job in self.jobs:
            if job.process.poll() is None:
                os.killpg(job.process.pid, signal.SIGINT)
        for job in self.jobs:
            try:
                job.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(job.process.pid, signal.SIGKILL)
        say("\nStopped. Start the same command again to continue: finished runs are skipped and interrupted "
            "runs resume from their last save.")
        sys.exit(130)


def write_summary() -> None:
    rows = []
    for path in sorted(OUT.glob("**/metrics.json")):
        metrics = json.loads(path.read_text())
        points = metrics["curves"]["deterministic"]["points"]
        for mode, skills in metrics["summary"].items():
            for skill, values in skills.items():
                rows.append({"group": path.relative_to(OUT).parts[0], "run": path.parent.name, "mode": mode,
                             "skill": skill, "last_step": points[-1]["step"] if points else None, **values})
    if rows:
        with open(OUT / "summary.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            writer.writeheader()
            writer.writerows(rows)


def describe_experiments() -> None:
    """--list: every experiment's description and settings."""
    for experiment in EXPERIMENTS.values():
        say(f"{experiment.name}\n{textwrap.indent(inspect.getdoc(experiment), '    ')}\n    settings:")
        for name, value in experiment.settings().items():
            say(f"      {name}={setting_text(value)}")
        say()


def main() -> None:
    if ARGUMENTS.list:
        describe_experiments()
        return
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    for condition in OPERATOR_EMBEDDINGS:
        condition_options(condition)
    check_repair_modes()
    check_operator_names()
    check_repairs()
    check_orders()
    overlap = overlap_report(BASE_SKILLS, REPAIRED)  # how much of their operators the skills share (static)
    if not RUN_GROUPS:  # a static experiment (overlap): the report is all it does
        say(overlap)
        return
    runs = build_plan()
    if not runs:
        sys.exit(f"nothing to run for RUN_GROUPS={' '.join(RUN_GROUPS)}")
    runner = Runner(runs)
    say(f"Experiment {EXPERIMENT.name} · {datetime.now():%Y-%m-%d %H:%M} · groups: {', '.join(RUN_GROUPS)}")
    say(f"  {inspect.getdoc(EXPERIMENT).splitlines()[0]}")
    if OVERRIDES:
        say("  overridden by the environment: " + "  ".join(f"{name}={value}" for name, value in OVERRIDES.items()))
    say(f"  base skills: {', '.join(BASE_SKILLS)}")
    named = [condition for condition in OPERATOR_EMBEDDINGS if name_excluded_options(condition)]
    if named and {"rq1", "repair"} & set(RUN_GROUPS):
        choices = {"exclude": "excluded (<condition>-no-name)", "include": "included (<condition>)"}
        say(f"  operator name in {', '.join(named)}: {' and '.join(choices[choice] for choice in OPERATOR_NAMES)}")
    if "repair" in RUN_GROUPS:
        for repair_from, repair_to in REPAIRS:
            say(f"  repair: {repair_from} -> {repair_to} ({repair_difference(repair_from, repair_to)})")
    if "rq2" in RUN_GROUPS:
        say("  rq2 orders, each added skill with its literal coverage by the skills before it (zero-shot needs 1):")
        for order in RQ2_ORDERS:
            coverage = [f"{skill} {value:.2f}" for number in range(2, len(order) + 1)
                        for skill, value in stage_coverage(order, number).items()]
            say(f"    {order_name(order)}: {', '.join(coverage) or 'nothing added'}")
    say(overlap)
    runner.print_plan()
    if DRY_RUN:
        for run in runs:
            say(f"\n{run.label}:\n  {PYTHON} {' '.join(run.command)}")
        say("\nDRY_RUN=1: nothing started")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "overnight.lock"
    if lock.exists():
        pid = int(lock.read_text() or 0)
        try:
            os.kill(pid, 0)
            sys.exit(f"another overnight run (pid {pid}) is using {OUT}")
        except (OSError, ValueError):
            pass  # stale lock from a run that is gone
    lock.write_text(str(os.getpid()))
    (OUT / "overlap.txt").write_text(overlap + "\n")
    signal.signal(signal.SIGINT, runner.interrupt)
    signal.signal(signal.SIGTERM, runner.interrupt)
    try:
        say()
        preflight(runs)
        runner.execute()
        write_summary()
        subprocess.run([PYTHON, "scripts/review_runs.py", str(OUT)], cwd=ROOT, capture_output=True)
        say(f"\nAll done in {duration(time.monotonic() - runner.started)}. "
            f"Failed: {', '.join(sorted(runner.failed - runner.ended)) or 'none'}"
            + (f". Not run, their chain having ended: {', '.join(sorted(runner.ended))}" if runner.ended else ""))
        say(f"  report:  {OUT / 'review.txt'}\n  table:   {OUT / 'summary.csv'}")
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
