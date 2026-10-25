#!/usr/bin/env python3
"""Condense a folder of training runs (e.g. scripts/overnight_runs.py output) into a text report to review.

    ./scripts/review_runs.py outputs/overnight_20261008            # print the report and save <folder>/review.txt
    ./scripts/review_runs.py outputs/overnight_20261008 --no-embeddings

Sections:
    1. runs         every run: finished or not, last step, stop reason, why an rq2 chain ended before its
                    last stage, warnings / errors in its log
    2. conditions   per condition (runs differing only in seed): mean ± std over seeds, with a 95% confidence
                    interval (CI) of the mean via bootstrap resampling of the seeds' values (no run is
                    repeated), of steps and episodes to threshold, AUC and final success
                    rate, per skill, deterministic and stochastic
    3. curves       per run and skill: deterministic success rate at evenly spaced evaluations
    4. training     per run: start -> end of key Stable-Baselines3 statistics from TensorBoard (rewards,
                    entropy / action std, KL, explained variance, losses) and the operator embedding
                    magnitudes logged during training, flagging NaN or exploding values
    5. added skills per run where a skill was added or replaced after training started (repair / new
                    skill runs; an rq2 chain's run has one per stage after the first): the step it was added,
                    its zero-shot success rate there, and the steps to threshold from there, per condition (as
                    in section 2); and what the other skills forgot: their mean level before, lowest after and
                    final, and the steps until all were back at their level
    6. embeddings   per run with a learned operator embedding (trainable or compositional): the final
                    embedding of each skill, from the saved model: its magnitude (L2 norm, i.e. length)
                    and the distance and cosine similarity between skills (low separation means the
                    policy sees skills as alike)
"""

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

LOG_PATTERNS = {"errors": r"Traceback|Error", "warnings": r" WARNING ", "setup failures": r"setup failed"}
TRAINING_TAGS = [
    "rollout/ep_rew_mean", "rollout/success_rate", "train/entropy_loss", "train/std", "train/approx_kl",
    "train/explained_variance", "train/value_loss", "train/clip_fraction", "train/ent_coef", "train/critic_loss",
    "train/actor_loss",
]


def censored(values: list, trained: list) -> tuple[list, int]:
    """A time to reach something over seeds (e.g. steps to threshold), with each seed that never got there (None)
    counted as what it trained (trained: each seed's training experience), and the number of such seeds. Left out,
    those seeds would make a condition that often fails look fast. Counted so, each is at most its true time, so the
    mean, both ends of its bootstrap 95% CI (each resample's mean) and a transfer gain from them are lower bounds,
    as rq1_budget in overnight_runs.py counts a skill that never reached the threshold as all its steps."""
    never = sum(value is None and steps is not None for value, steps in zip(values, trained))
    return [steps if value is None else value for value, steps in zip(values, trained)], never


def mean_std(values: list, trained: list | None = None) -> str:
    """Mean ± std over seeds and, from 2 seeds on, the 95% confidence interval (CI) of the mean via bootstrap
    resampling (bootstrap_interval), printed [95% CI via bootstrap: low, high]; seeds without a value are left
    out. With trained, a time to reach something (see censored): ≥ marks lower
    bounds, followed by the number of seeds that never got there."""
    values, never = censored(values, trained) if trained is not None else (values, 0)
    values = [v for v in values if v is not None]
    if len(values) == never:
        return "never"
    if len(values) == 1:
        return f"{values[0]:.3g}"
    low, high = bootstrap_interval(values)
    return (f"{'≥' if never else ''}{statistics.mean(values):.3g} ± {statistics.stdev(values):.2g}"
            f" [95% CI via bootstrap: {low:.3g}, {high:.3g}]{f' ({never} never)' if never else ''}")


def bootstrap_interval(values: list, confidence: float = 0.95, resamples: int = 10_000) -> tuple[float, float]:
    """95% confidence interval (CI) of the mean via percentile bootstrap resampling: the middle `confidence` of the
    means of `resamples` resamples (with replacement) of values. It resamples the numbers already collected, one per
    seed; no run is repeated. The resamples are drawn from a fixed seed so the same runs always give the same report.
    It never reaches beyond the values' range, and with 3 values or fewer it is that range, which holds the true
    mean less often than `confidence` says (3 times in 4 for 3 values from a symmetric distribution)."""
    import numpy as np

    values = np.asarray(values, dtype=float)
    means = values[np.random.default_rng(0).integers(len(values), size=(resamples, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [(1 - confidence) / 2, (1 + confidence) / 2])
    return float(low), float(high)


def training_experience(metrics: dict, skill: str, measure: str = "steps") -> int | None:
    """Training experience in measure (steps, episodes or skill_steps) from the skill's first evaluation to its
    last: what a run that never reached the threshold trained without reaching it (None if not recorded)."""
    points = [point for point in metrics["curves"]["deterministic"]["points"] if skill in point["result"]]
    if not points:
        return None
    if measure == "steps":
        return points[-1]["step"] - points[0]["step"]
    first, last = points[0].get("progress"), points[-1].get("progress")
    if not (first and last):
        return None
    if measure == "episodes":
        return last["episodes"] - first["episodes"]
    return last[measure].get(skill, 0) - first[measure].get(skill, 0)


def condition_name(run: Path) -> str:
    """The run's folder and run name without their seeds: runs of one condition that differ only in seed
    (rq1/one-hot_seed0/<name>_seed0 and rq1/one-hot_seed1/<name>_seed1) share it."""
    return re.sub(r"_seed\d+(?=/|$)", "", f"{run.parent.name}/{run.name}")


def section_runs(runs: list[Path], folder: Path) -> list[str]:
    lines = ["1. RUNS", f"   {len(runs)} runs in {folder}"]
    failed = folder / "failed.txt"
    if failed.exists() and failed.read_text().strip():
        lines.append(f"   FAILED (failed.txt): {failed.read_text().split()}")
    for run in runs:
        metrics = load_metrics(run)
        log = (run / "train.log").read_text() if (run / "train.log").exists() else ""
        finished = "finished" if "training finished" in log else ("STOPPED" if "training stopped" in log else "UNFINISHED")
        last = metrics["curves"]["deterministic"]["points"][-1]["step"] if metrics else None
        counts = {name: len(re.findall(pattern, log)) for name, pattern in LOG_PATTERNS.items()}
        issues = ", ".join(f"{name} {count}" for name, count in counts.items() if count) or "no warnings"
        stop = f", stop: {metrics['stop_reason']}" if metrics and metrics.get("stop_reason") else ""
        ended = run / "chain_ended.txt"  # written by overnight_runs.py
        stop += f"; {ended.read_text().strip()}" if ended.exists() else ""
        lines.append(f"   {run.parent.name}/{run.name}: {finished}, last eval step {last}{stop}; {issues}")
    return lines


def load_metrics(run: Path) -> dict | None:
    path = run / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def section_conditions(runs: list[Path]) -> list[str]:
    lines = ["2. CONDITIONS (mean ± std over seeds [95% CI of the mean via bootstrap resampling of the seeds, from 2 seeds]; "
             "≥ ... (k never): k seeds never reached the threshold and count as what they trained, so the mean and "
             "CI are lower bounds; steps / episodes to threshold count from the skill's zero-shot point; "
             "on skill: of those steps, the ones spent on the skill itself)"]
    conditions: dict[str, list[dict]] = {}
    for run in runs:
        metrics = load_metrics(run)
        if metrics:
            conditions.setdefault(condition_name(run), []).append(metrics)
    for condition, metrics_list in sorted(conditions.items()):
        lines.append(f"   {condition}  ({len(metrics_list)} seeds)")
        for mode in ("deterministic", "stochastic"):
            skills = sorted({skill for metrics in metrics_list for skill in metrics["summary"][mode]})
            for skill in skills:
                values = [metrics["summary"][mode].get(skill, {}) for metrics in metrics_list]
                steps, skill_steps, episodes = (
                    mean_std([v.get(f"{measure}_to_threshold") for v in values],
                             [training_experience(metrics, skill, measure) for metrics in metrics_list])
                    for measure in ("steps", "skill_steps", "episodes"))
                lines.append(
                    f"     {mode[:5]} {skill:10s} steps {steps:>34s}"
                    f"  on skill {skill_steps:>34s}"
                    f"  episodes {episodes:>34s}"
                    f"  auc {mean_std([v.get('auc') for v in values]):>26s}"
                    f"  final {mean_std([v.get('final_success_rate') for v in values]):>24s}"
                )
    return lines


def section_curves(runs: list[Path], points_shown: int = 10) -> list[str]:
    lines = ["3. CURVES (deterministic success rate at evenly spaced evaluations: step -> rate)"]
    for run in runs:
        metrics = load_metrics(run)
        if not metrics:
            continue
        points = metrics["curves"]["deterministic"]["points"]
        indices = sorted({round(i * (len(points) - 1) / max(points_shown - 1, 1)) for i in range(points_shown)})
        for skill in sorted({skill for point in points for skill in point["result"]}):
            shown = [f"{points[i]['step'] // 1000}k:{points[i]['result'][skill]['success_rate']:.2f}"
                     for i in indices if skill in points[i]["result"]]
            lines.append(f"   {run.name} {skill}: {' '.join(shown)}")
    return lines


def section_training(runs: list[Path]) -> list[str]:
    lines = ["4. TRAINING (Stable-Baselines3 statistics from TensorBoard: first -> last logged value [min, max])"]
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return lines + ["   tensorboard is not installed; skipped"]
    for run in runs:
        folders = sorted((run / "tensorboard").glob("*"))
        if not folders:
            continue
        accumulator = EventAccumulator(str(folders[-1]), size_guidance={"scalars": 0})
        accumulator.Reload()
        available = set(accumulator.Tags()["scalars"])
        lines.append(f"   {run.name}")
        magnitude_tags = sorted(tag for tag in available if tag.startswith("operator_embedding_magnitude"))
        width = max([28, *(len(tag) for tag in magnitude_tags)])
        for tag in TRAINING_TAGS + magnitude_tags:
            if tag not in available:
                continue
            values = [event.value for event in accumulator.Scalars(tag)]
            flag = "  <-- NaN" if any(math.isnan(v) for v in values) else (
                "  <-- exploding" if max(abs(v) for v in values) > 1e6 else "")
            lines.append(f"     {tag:{width}s} {values[0]:10.4g} -> {values[-1]:10.4g}  [{min(values):.4g}, {max(values):.4g}]{flag}")
    return lines


def section_added_skills(runs: list[Path]) -> list[str]:
    """Skills first evaluated after step 0 (added on resume, e.g. a repaired operator replacing its
    original, or each later stage of an rq2 chain, several in one run): zero-shot success at the step they were added
    and steps to threshold from there, what the other skills forgot after that addition, and the transfer gain: the
    steps the skill needed from scratch (its learnability run in the same folder) divided by the steps it needed on
    itself after being added."""
    lines = ["5. ADDED SKILLS (added or replaced after training started: step added, zero-shot success there "
             "(deterministic / stochastic), steps to threshold from there; over seeds as in section 2)"]
    conditions: dict[str, list[tuple]] = {}
    from_scratch: dict[str, list] = {}  # skill -> (deterministic steps to threshold, steps trained) per learnability run
    for run in runs:
        metrics = load_metrics(run)
        if metrics and run.parent.parent.name == "learnability":
            from_scratch.setdefault(run.parent.name, []).append(
                (metrics["summary"]["deterministic"].get(run.parent.name, {}).get("steps_to_threshold"),
                 training_experience(metrics, run.parent.name)))
    for run in runs:
        metrics = load_metrics(run)
        if not metrics:
            continue
        points = metrics["curves"]["deterministic"]["points"]
        first_seen = {}
        for point in points:
            for skill in point["result"]:
                first_seen.setdefault(skill, point["step"])
        latest = set(points[-1]["result"]) if points else set()
        removed = sorted(skill for skill in first_seen if skill not in latest)
        for skill, step in first_seen.items():
            if step == 0:
                continue
            summary = {mode: metrics["summary"][mode].get(skill, {}) for mode in ("deterministic", "stochastic")}
            trained = {measure: training_experience(metrics, skill, measure) for measure in ("steps", "skill_steps")}
            # the retention of this addition: a run with several (an rq2 chain) keeps the earlier ones' apart
            retention = next((entry for entry in [*metrics.get("earlier_retention", []), metrics.get("retention")]
                              if entry and entry["added_at"] == step), None)
            conditions.setdefault(f"{condition_name(run)} {skill}", []).append(
                (step, summary, removed, retention, trained))
    # by condition, and a run's additions in the order they were added
    for key, entries in sorted(conditions.items(), key=lambda item: (item[0].rsplit(" ", 1)[0], item[1][0][0])):
        removed = entries[0][2]
        lines.append(f"   {key}  ({len(entries)} seeds; added at step {mean_std([e[0] for e in entries])}"
                     f"{'; replaced ' + ', '.join(removed) if removed else ''})")
        zero_shot = [e[1]["deterministic"].get("zero_shot_success_rate") for e in entries]
        if "repair-hidden" in key and zero_shot and statistics.mean(v or 0 for v in zero_shot) >= 0.5:
            lines.append("     <-- REPAIR INVALID: the policy succeeds without seeing the repair (repair-hidden), so the "
                         "original behaviour already satisfies the repaired operator")
        for mode in ("deterministic", "stochastic"):
            values = [e[1][mode] for e in entries]
            steps, skill_steps = (
                mean_std([v.get(f"{measure}_to_threshold") for v in values], [e[4][measure] for e in entries])
                for measure in ("steps", "skill_steps"))
            lines.append(f"     {mode[:5]}  zero-shot {mean_std([v.get('zero_shot_success_rate') for v in values]):>24s}"
                         f"  steps to threshold {steps:>34s}"
                         f"  on skill {skill_steps:>34s}"
                         f"  final {mean_std([v.get('final_success_rate') for v in values]):>24s}")
        skill = key.rsplit(" ", 1)[1]
        if skill in from_scratch:
            alone, never = censored(*zip(*from_scratch[skill]))
            alone = [steps for steps in alone if steps is not None]
            after = [e[1]["deterministic"].get("skill_steps_to_threshold") for e in entries]
            if len(alone) > never and after and all(steps is not None for steps in after):
                gain = ("zero-shot" if statistics.mean(after) == 0 else
                        f"{'≥' if never else ''}{statistics.mean(alone) / statistics.mean(after):.2f}x "
                        "(from scratch / after the repair)")
            else:
                gain = "n/a: " + ("never reached from scratch" if len(alone) == never else "not reached after the repair")
            lines.append(f"     from scratch (learnability)  steps to threshold {mean_std(*zip(*from_scratch[skill])):>34s}"
                         f"  transfer gain {gain}")
        retentions = [(e[3], e[4]["steps"]) for e in entries if e[3] and e[3]["skills"]]
        if retentions:
            def mean_of(key: str) -> list:
                return [statistics.mean(s[key] or 0 for s in r["skills"].values()) for r, _ in retentions]
            back = [r["all_recovered_at"] - r["added_at"] if r["all_recovered_at"] is not None else None
                    for r, _ in retentions]
            lines.append(f"     others (mean over skills)  before {mean_std(mean_of('level_before')):>24s}"
                         f"  lowest after {mean_std(mean_of('lowest_after')):>24s}"
                         f"  final {mean_std(mean_of('final')):>24s}"
                         f"  steps until all back {mean_std(back, [steps for _, steps in retentions]):>34s}")
    if len(lines) == 1:
        lines.append("   none")
    return lines


def section_embeddings(runs: list[Path]) -> list[str]:
    lines = ["6. EMBEDDINGS (final learned operator embedding per skill: magnitude (L2 norm); pairwise distance and cosine)"]
    for run in runs:
        config_path, model_path = run / "config.json", run / "model.zip"
        if not (config_path.exists() and model_path.exists()):
            continue
        config = json.loads(config_path.read_text())
        if not (config.get("trainable_operator_embedding") or config.get("operator_encoder") == "compositional"):
            continue
        try:
            embeddings = learned_embeddings(config, model_path)
        except Exception as error:  # report and continue: one broken run should not stop the review
            lines.append(f"   {run.name}: could not compute ({type(error).__name__}: {error})")
            continue
        names = list(embeddings)
        magnitudes = ", ".join(f"{name} {embeddings[name].norm():.3g}" for name in names)
        pairs = []
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                a, b = embeddings[first], embeddings[second]
                cosine = float(a @ b / (a.norm() * b.norm() + 1e-12))
                pairs.append(f"{first}-{second} distance {(a - b).norm():.3g} cosine {cosine:.3f}")
        lines.append(f"   {run.name}: magnitudes {magnitudes}; {'; '.join(pairs)}")
        if config.get("operator_encoder") == "compositional":
            raw = composition_magnitudes(config, model_path)
            lines.append(f"     before normalization: magnitudes {', '.join(f'{name} {value:.3g}' for name, value in raw.items())}")
    return lines


def learned_embeddings(config: dict, model_path: Path) -> dict:
    """Each skill's operator embedding as the saved policy computes it (encoder, then the learned extractor)."""
    import numpy as np
    import torch

    from cam.experiments.environment_setup import OPERATOR_ENCODERS
    from cam.policies.feature_extractors import features_extractors
    from cam.policies.stable_baselines3_policy import ALGORITHMS
    from cam.skills.registry import build_skill

    skills = [build_skill(name, config) for name in config["skills"]]
    encoder = OPERATOR_ENCODERS[config["operator_encoder"]](config, skills)
    inputs = torch.as_tensor(np.stack([encoder.encode(skill.symbolic_action_model) for skill in skills]))
    policy = ALGORITHMS[config["algorithm"]].load(model_path, device="cpu").policy
    extractor = features_extractors(policy)["actor"]
    with torch.no_grad():
        outputs = extractor.operator_embedding(inputs)
    return {skill.name: output for skill, output in zip(skills, outputs)}


def composition_magnitudes(config: dict, model_path: Path) -> dict:
    """Compositional runs only: each skill's composed embedding magnitude (L2 norm) before the extractor
    normalizes it. The normalized embedding always has magnitude 1, so this is where a growing composition
    still shows (and it is what policies trained before the normalization was added actually saw)."""
    import numpy as np
    import torch

    from cam.experiments.environment_setup import OPERATOR_ENCODERS
    from cam.policies.feature_extractors import features_extractors
    from cam.policies.stable_baselines3_policy import ALGORITHMS
    from cam.skills.registry import build_skill

    skills = [build_skill(name, config) for name in config["skills"]]
    encoder = OPERATOR_ENCODERS[config["operator_encoder"]](config, skills)
    inputs = torch.as_tensor(np.stack([encoder.encode(skill.symbolic_action_model) for skill in skills]))
    policy = ALGORITHMS[config["algorithm"]].load(model_path, device="cpu").policy
    extractor = features_extractors(policy)["actor"]
    with torch.no_grad():
        composed = extractor.composed_embedding(inputs)
    return {skill.name: float(magnitude) for skill, magnitude in zip(skills, composed.norm(dim=1))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder of runs, e.g. outputs/overnight_<date>")
    parser.add_argument("--no-embeddings", action="store_true", help="skip section 6 (loads every model)")
    args = parser.parse_args()
    runs = sorted(path.parent for path in args.folder.glob("**/config.json") if (path.parent / "metrics.json").exists())
    if not runs:
        sys.exit(f"no runs with config.json and metrics.json under {args.folder}")
    sections = [
        section_runs(runs, args.folder), section_conditions(runs), section_curves(runs), section_training(runs),
        section_added_skills(runs),
    ]
    if not args.no_embeddings:
        sections.append(section_embeddings(runs))
    report = "\n\n".join("\n".join(section) for section in sections)
    print(report)
    (args.folder / "review.txt").write_text(report + "\n")
    print(f"\nsaved {args.folder / 'review.txt'}")


if __name__ == "__main__":
    main()
