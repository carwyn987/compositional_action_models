#!/usr/bin/env python3
"""Overnight experiments for the research questions: a plan of training runs, executed and explained.

    ./scripts/start_overnight.sh                     # usual way: environment checks + unit tests, then this
    .venv/bin/python scripts/overnight_runs.py       # this runner alone
    DRY_RUN=1 .venv/bin/python scripts/overnight_runs.py              # print the plan, run nothing
    RUN_GROUPS="learnability" SEEDS="0 1" JOBS=6 .venv/bin/python scripts/overnight_runs.py

How it runs
  * Up to JOBS runs at once (default 2). One run uses a fraction of a GPU or CPU: SAC alternates a
    simulator step (CPU) with one small gradient step, so several runs side by side get through the plan
    several times faster. JOBS=1 runs one at a time.
  * Each run is a child process (main.py) whose full output goes to <OUT>/logs/<label>.log. The terminal
    shows, as each run starts, what it trains and why (which research question or check it serves);
    while runs go, their progress (steps, steps/s, time left) and the overall time left; when one ends,
    its outcome. A repair run waits for the RQ1 run it continues.
  * Ctrl+C stops every running run after it saves. Starting the same command again skips finished runs and
    resumes interrupted ones from their last save.
  * Before anything long starts (fail fast): the text backend is checked and every operator text the runs
    need is embedded and cached (TEXT_BACKEND=openai needs the API key and network), the device is reported,
    and every skill's scripted policy must succeed at least once in 3 episodes (which checks each skill's
    setup and predicates).

Groups
  learnability  each skill alone (BASE_SKILLS and every repaired skill), multi-hot: can it be trained at
                all, and how fast? Also the "individual policy" baseline for RQ4 (hypernetworks).
  rq1           RQ1: one shared policy on BASE_SKILLS for each operator embedding condition (CONDITIONS)
                and seed: steps to threshold, AUC and final success per condition.
  repair        RQ2 / RQ3: for each FROM:TO in REPAIRS, each rq1 run continued with FROM replaced by its
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
                  repair-hidden   (ALIAS_CONDITIONS only) the repaired operator gets exactly FROM's
                                  embedding (--operator-identity-aliases), so the policy cannot see the
                                  repair. A control: if it succeeds zero-shot, the repair needs no new
                                  behaviour and the test is invalid (review.txt flags it).

Outputs, under OUT (default outputs/overnight_<date>)
  learnability/<skill>/<run name>/, rq1/<condition>_seed<n>/<run name>/   run directories (metrics.json,
                                                                            model.zip, tensorboard/, ...)
  repair/<TO>/<condition>_<repair-visible|repair-hidden>_seed<n>/          copy of the rq1 run, continued
  logs/<label>.log   each run's full output        failed.txt   labels of runs that failed
  summary.csv        per run and skill metrics     review.txt   scripts/review_runs.py's report
View the curves with: LOGDIR=<OUT> ./scripts/start_tensorboard.sh

Settings (environment variables, defaults in brackets)
  OUT [outputs/overnight_<date>]  RUN_GROUPS ["learnability rq1 repair"]  ALGORITHM [sac]  SEEDS ["0"]
  JOBS [2] runs at once   DEVICE [auto]   THREADS [cores / JOBS] torch threads per run   DRY_RUN [0]
  CONDITIONS [one-hot random multi-hot-trainable text compositional-tree compositional-slots
              compositional-geometric]   ALIAS_CONDITIONS ["one-hot random"]
  TEXT_BACKEND [openai] (the text condition and compositional name embeddings; mock: offline, no meaning)
  BASE_SKILLS [pickup putdown stack unstack pickup-raised push-left push-right push-front push-back
               push-right-front push-left-back push-right-back push-beside]
  REPAIRS ["unstack:unstack-raised stack:place-beside push-left:push-left-front"] (FROM:TO pairs; every
          literal of each TO appears in some base skill, so a policy reading the operator can use it)
  REPAIR_MODES ["isolated retain"]  REPAIR_EVAL_INTERVAL [10000] (how often a repair run evaluates every skill)
  LEARN_STEPS [150000]  RQ1_STEPS [500000]  REPAIR_STEPS [150000] (a repair run's most steps)
  EVAL_INTERVAL [50000]  EVAL_EPISODES [20]  SAVE_INTERVAL [50000]
"""

import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").exists() else sys.executable


def setting(name: str, default: str) -> str:
    return os.environ.get(name, default)


OUT = Path(setting("OUT", f"outputs/overnight_{date.today():%Y%m%d}"))
RUN_GROUPS = setting("RUN_GROUPS", "learnability rq1 repair").split()
ALGORITHM = setting("ALGORITHM", "sac")
SEEDS = [int(seed) for seed in setting("SEEDS", "0").split()]
CONDITIONS = setting(
    "CONDITIONS",
    "one-hot random multi-hot-trainable text compositional-tree compositional-slots compositional-geometric",
).split()
ALIAS_CONDITIONS = setting("ALIAS_CONDITIONS", "one-hot random").split()
TEXT_BACKEND = setting("TEXT_BACKEND", "openai")
BASE_SKILLS = setting(  # the training skills; held-out operators (REPAIRS' TO) only appear in repair runs
    "BASE_SKILLS",
    "pickup putdown stack unstack pickup-raised push-left push-right push-front push-back "
    "push-right-front push-left-back push-right-back push-beside",
).split()
REPAIRS = [tuple(pair.split(":")) for pair in
           setting("REPAIRS", "unstack:unstack-raised stack:place-beside push-left:push-left-front").split()]
REPAIRED = [to for _, to in REPAIRS]
REPAIR_MODES = setting("REPAIR_MODES", "isolated retain").split()
REPAIR_EVAL_INTERVAL = int(setting("REPAIR_EVAL_INTERVAL", "10000"))
LEARN_STEPS = int(setting("LEARN_STEPS", "150000"))
RQ1_STEPS = int(setting("RQ1_STEPS", "500000"))
REPAIR_STEPS = int(setting("REPAIR_STEPS", "150000"))
EVAL_INTERVAL = int(setting("EVAL_INTERVAL", "50000"))
EVAL_EPISODES = int(setting("EVAL_EPISODES", "20"))
SAVE_INTERVAL = int(setting("SAVE_INTERVAL", "50000"))
DEVICE = setting("DEVICE", "auto")
JOBS = max(1, int(setting("JOBS", "2")))
THREADS = setting("THREADS", str(max(1, (os.cpu_count() or 1) // JOBS)))  # no CPU oversubscription
DRY_RUN = setting("DRY_RUN", "0") == "1"
STATUS_EVERY = 60 if sys.stdout.isatty() else 600  # seconds between progress reports while runs go

# Operator embedding conditions: name -> (main.py options, what the policy is told about the skill).
CONDITION_OPTIONS = {
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
# Rough single-run SAC steps/s per condition on a CPU, for the first time estimate; replaced by the speeds
# measured as runs finish.
EXPECTED_RATE = {"compositional-tree": 24, "compositional-slots": 23, "compositional-geometric": 18}
DEFAULT_RATE = 40
COMMON = [
    "--algorithm", ALGORITHM, "--reward", "shaped", "--num-blocks", "2", "--device", DEVICE,
    "--eval-interval", str(EVAL_INTERVAL), "--eval-episodes", str(EVAL_EPISODES),
    "--save-interval", str(SAVE_INTERVAL), "--text-backend", TEXT_BACKEND,
]
PROGRESS = re.compile(r"(\d[\d,]*)/(\d[\d,]*)\s")


@dataclass
class Run:
    label: str
    group: str
    title: str
    what: str
    why: str
    steps: int  # training steps this run adds
    condition: str  # for the time estimate
    output: Path  # new runs: --output-dir; repair runs: the run directory itself
    command: list[str]
    copy_from: Path | None = None  # repair runs: the rq1 output directory holding the run to continue
    depends_on: str | None = None
    repaired: str | None = None  # repair runs: the repaired skill

    @property
    def log(self) -> Path:
        return OUT / "logs" / f"{self.label}.log"

    def finished(self) -> bool:
        return self.log.exists() and "training finished" in self.log.read_text(errors="replace")

    def run_directory(self) -> Path | None:
        if self.copy_from is not None:
            return self.output if (self.output / "config.json").exists() else None
        found = sorted(path.parent for path in self.output.glob("*/config.json"))
        return found[0] if found else None


def condition_options(condition: str) -> list[str]:
    if condition not in CONDITION_OPTIONS:
        sys.exit(f"unknown condition {condition!r}; known: {', '.join(CONDITION_OPTIONS)}")
    return CONDITION_OPTIONS[condition][0]


def repair_difference(repair_from: str, repair_to: str) -> str:
    """What repair_to changes relative to repair_from, e.g. 'adds effect (raised ?o)'."""
    from cam.skills.registry import build_skill

    config = {"symbolic_action_model_format": "pddl"}
    def literals(name: str) -> set[str]:
        model = build_skill(name, config).symbolic_action_model
        return {f"precondition {'(not ' if p.negated else ''}{p.predicate}{')' if p.negated else ''}"
                for p in model.preconditions} | {
                f"effect {'(not ' if e.delete else ''}{e.predicate}{')' if e.delete else ''}" for e in model.effects}
    before, after = literals(repair_from), literals(repair_to)
    changes = [f"adds {literal}" for literal in sorted(after - before)]
    changes += [f"removes {literal}" for literal in sorted(before - after)]
    return ", ".join(changes) or "no literal changes"


def build_plan() -> list[Run]:
    runs: list[Run] = []
    skill_list = ", ".join(BASE_SKILLS)
    if "learnability" in RUN_GROUPS:
        for skill in dict.fromkeys(BASE_SKILLS + REPAIRED):
            output = OUT / "learnability" / skill
            runs.append(Run(
                label=f"learnability_{skill}", group="learnability", title=f"Learnability · {skill}",
                what=f"one {ALGORITHM.upper()} policy trained on {skill} alone (multi-hot embedding), {LEARN_STEPS:,} steps",
                why=(f"can {skill} be learned at all, and how fast? If not, no result that uses it means anything. "
                     "Also the single-skill baseline for RQ4 (hypernetworks vs individual policies)."),
                steps=LEARN_STEPS, condition="multi-hot", output=output,
                command=["main.py", "--skills", skill, "--operator-encoder", "multi-hot",
                         "--total-timesteps", str(LEARN_STEPS), "--seed", "0", *COMMON, "--output-dir", str(output)],
            ))
    for seed in SEEDS:
        for condition in CONDITIONS:
            options = condition_options(condition)
            description = CONDITION_OPTIONS[condition][1]
            rq1_label, rq1_output = f"rq1_{condition}_seed{seed}", OUT / "rq1" / f"{condition}_seed{seed}"
            if "rq1" in RUN_GROUPS:
                runs.append(Run(
                    label=rq1_label, group="rq1", title=f"RQ1 · {condition} · seed {seed}",
                    what=(f"one shared {ALGORITHM.upper()} policy on {len(BASE_SKILLS)} skills ({skill_list}), "
                          f"{RQ1_STEPS:,} steps; the policy is told which skill to do through {description}"),
                    why=("RQ1: does conditioning on this representation of the operator make multi-skill learning "
                         "faster or better than an arbitrary skill ID? Compared across conditions on steps to "
                         "threshold, AUC and final success."),
                    steps=RQ1_STEPS, condition=condition, output=rq1_output,
                    command=["main.py", "--skills", *BASE_SKILLS, *options, "--total-timesteps", str(RQ1_STEPS),
                             "--seed", str(seed), *COMMON, "--output-dir", str(rq1_output)],
                ))
            if "repair" not in RUN_GROUPS:
                continue
            for (repair_from, repair_to), mode in ((pair, mode) for pair in REPAIRS for mode in REPAIR_MODES):
                difference = repair_difference(repair_from, repair_to)
                skills = [repair_to if skill == repair_from else skill for skill in BASE_SKILLS]
                variants = ["repair-visible"] + (["repair-hidden"] if condition in ALIAS_CONDITIONS else [])
                trains = (f"trains only {repair_to} until it reaches the success threshold (the other skills are "
                          "evaluated, not trained)" if mode == "isolated" else
                          f"trains every skill, as the RQ1 run did, until {repair_to} reaches the success threshold "
                          "and every other skill is back at its level before the repair")
                for variant in variants:
                    run_directory = OUT / "repair" / repair_to / f"{condition}_{variant}_{mode}_seed{seed}"
                    hidden = variant == "repair-hidden"
                    runs.append(Run(
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
                        steps=REPAIR_STEPS, condition=condition, output=run_directory, copy_from=rq1_output,
                        depends_on=rq1_label if "rq1" in RUN_GROUPS else None, repaired=repair_to,
                        command=["main.py", "--run-directory", str(run_directory), "--skills", *skills,
                                 "--stop-skills", repair_to, "--repair-mode", mode,
                                 "--total-timesteps", str(REPAIR_STEPS), "--eval-interval", str(REPAIR_EVAL_INTERVAL),
                                 "--device", DEVICE,
                                 *(["--operator-identity-aliases", f"{repair_to}={repair_from}"] if hidden else [])],
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
    skills = sorted(set(BASE_SKILLS) | (set(REPAIRED) if uses_repair_skill else set()))
    if any(run.condition == "text" or run.condition.startswith("compositional") for run in runs):
        check_text_backend(skills)
    check_scripted_policies(skills)
    free = shutil.disk_usage(OUT if OUT.exists() else ROOT).free / 1e9
    say(f"  ✓ free disk space: {free:.0f} GB" + ("  (low: runs save models and logs)" if free < 10 else ""))


def check_repair_modes() -> None:
    unknown = [mode for mode in REPAIR_MODES if mode not in ("isolated", "retain")]
    if unknown:
        sys.exit(f"unknown REPAIR_MODES {unknown}; known: isolated, retain")


def check_text_backend(skills: list[str]) -> None:
    """Embed every operator text the runs need now, through the same cache the runs use, so a missing key
    or network fails here and training never waits on the API."""
    from cam.experiments.environment_setup import OPERATOR_ENCODERS, text_backend
    from cam.skills.registry import build_skill
    from main import parse_args

    config = dict(parse_args(["--skills", *skills, "--text-backend", TEXT_BACKEND, "--operator-text", "paragraph"]))
    try:
        operators = [build_skill(skill, config) for skill in skills]
        encoder = OPERATOR_ENCODERS["text"](config, operators)
        names = text_backend(config, config["component_embedding_dim"])
        for skill in operators:
            encoder.encode(skill.symbolic_action_model)  # paragraph embedding (text condition)
            names.embed_text(skill.name)  # name embedding (compositional conditions)
    except Exception as error:  # report any failure (no key, no network, no openai package) and stop
        fail(f"text backend {TEXT_BACKEND!r} is not usable: {type(error).__name__}: {error}\n"
             "  set OPENAI_API_KEY or put the key in ~/.secrets/openai_api_key, or run with TEXT_BACKEND=mock")
    say(f"  ✓ text backend {TEXT_BACKEND}: {len(operators)} operator paragraphs and names embedded and cached "
        f"({config['text_embedding_cache']})")


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
        self.started = time.monotonic()
        self.jobs: list[Job] = []
        self.last_status = 0.0

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
            after = f"  (after {run.depends_on})" if run.depends_on else ""
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
                if run.depends_on and run.depends_on not in self.done | self.failed:
                    continue  # waits for the run it continues
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
        if run.depends_on in self.failed:
            say(f"   ⤼ skipped: {run.depends_on} failed")
            self.failed.add(run.label)
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

    def prepare(self, run: Run) -> list[str] | None:
        """The command to run: a new run, a repair copied from its rq1 run, or a resume of an interrupted run."""
        run_directory = run.run_directory()
        if run.copy_from is not None and run_directory is None:
            sources = sorted(path.parent for path in run.copy_from.glob("*/model.zip"))
            if not sources:
                say(f"   ✗ no finished rq1 run in {run.copy_from} to continue; skipped")
                self.failed.add(run.label)
                return None
            shutil.copytree(sources[0], run.output)
            base = json.loads((run.output / "metrics.json").read_text())["progress"]["steps"]
            (run.output / "repair_base_steps.txt").write_text(str(base))
            return run.command
        if run_directory is not None and (run_directory / "model.zip").exists():  # interrupted earlier
            done = json.loads((run_directory / "metrics.json").read_text())["progress"]["steps"]
            base = int((run_directory / "repair_base_steps.txt").read_text()) if run.copy_from else 0
            left = run.steps - (done - base)
            if left <= 0:
                say("   ✓ already at its step budget")
                return None
            say(f"   resuming an interrupted run at step {done:,}: {left:,} steps left")
            command = list(run.command)
            if run.copy_from is None:  # new run -> resume it in place
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
        """(steps done in this invocation, its total) from the progress bar at the end of the run's log."""
        try:
            with open(run.log, "rb") as log:
                log.seek(max(log.seek(0, 2) - 4000, 0))
                tail = log.read().decode(errors="replace").replace("\r", "\n")
        except OSError:
            return None, run.steps
        matches = PROGRESS.findall(tail)
        if not matches:
            return None, run.steps
        done, total = (int(value.replace(",", "")) for value in matches[-1])
        return done, total

    @staticmethod
    def report(run: Run) -> None:
        """The run's outcome from its metrics.json: per skill, final success and steps to threshold."""
        run_directory = run.run_directory()
        if run_directory is None or not (run_directory / "metrics.json").exists():
            return
        metrics = json.loads((run_directory / "metrics.json").read_text())
        summary = metrics["summary"]["deterministic"]
        if run.group == "repair":
            values = summary.get(run.repaired, {})
            say(f"   result: {run.repaired} zero-shot {values.get('zero_shot_success_rate')}, "
                f"steps to threshold {values.get('steps_to_threshold')}"
                + (f"; stopped: {metrics['stop_reason']}" if metrics.get("stop_reason") else ""))
            retention = metrics.get("retention") or {}
            others = retention.get("skills") or {}
            if others:
                mean = lambda key: sum(entry[key] or 0 for entry in others.values()) / len(others)  # noqa: E731
                back = retention.get("all_recovered_at")
                say(f"   other skills (mean): {mean('level_before'):.2f} before -> lowest {mean('lowest_after'):.2f} "
                    f"-> {mean('final'):.2f} at the end; "
                    + (f"all back after {back - retention['added_at']:,} steps" if back is not None else "not all back"))
            return
        cells = [f"{skill} {values.get('final_success_rate')}"
                 + (f" (95% at {values['steps_to_threshold'] // 1000}k)" if values.get("steps_to_threshold") is not None else "")
                 for skill, values in summary.items()]
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


def main() -> None:
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    for condition in CONDITIONS:
        condition_options(condition)
    check_repair_modes()
    runs = build_plan()
    if not runs:
        sys.exit(f"nothing to run for RUN_GROUPS={' '.join(RUN_GROUPS)}")
    runner = Runner(runs)
    say(f"Overnight experiments · {datetime.now():%Y-%m-%d %H:%M} · groups: {', '.join(RUN_GROUPS)}")
    say(f"  base skills: {', '.join(BASE_SKILLS)}")
    if "repair" in RUN_GROUPS:
        for repair_from, repair_to in REPAIRS:
            say(f"  repair: {repair_from} -> {repair_to} ({repair_difference(repair_from, repair_to)})")
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
    signal.signal(signal.SIGINT, runner.interrupt)
    signal.signal(signal.SIGTERM, runner.interrupt)
    try:
        say()
        preflight(runs)
        runner.execute()
        write_summary()
        subprocess.run([PYTHON, "scripts/review_runs.py", str(OUT)], cwd=ROOT, capture_output=True)
        say(f"\nAll done in {duration(time.monotonic() - runner.started)}. "
            f"Failed: {', '.join(sorted(runner.failed)) or 'none'}")
        say(f"  report:  {OUT / 'review.txt'}\n  table:   {OUT / 'summary.csv'}")
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
