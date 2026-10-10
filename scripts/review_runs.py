#!/usr/bin/env python3
"""Condense a folder of training runs (e.g. scripts/overnight_runs.sh output) into a text report to review.

    ./scripts/review_runs.py outputs/overnight_20261008            # print the report and save <folder>/review.txt
    ./scripts/review_runs.py outputs/overnight_20261008 --no-embeddings

Sections:
    1. runs         every run: finished or not, last step, stop reason, warnings / errors in its log
    2. conditions   per condition (runs differing only in seed): mean ± std over seeds of steps and episodes
                    to threshold, AUC and final success rate, per skill, deterministic and stochastic
    3. curves       per run and skill: deterministic success rate at evenly spaced evaluations
    4. training     per run: start -> end of key Stable-Baselines3 statistics from TensorBoard (rewards,
                    entropy / action std, KL, explained variance, losses) and the operator embedding
                    magnitudes logged during training, flagging NaN or exploding values
    5. embeddings   per run with a learned operator embedding (trainable or compositional): the final
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


def mean_std(values: list) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "never"
    if len(values) == 1:
        return f"{values[0]:.3g}"
    return f"{statistics.mean(values):.3g} ± {statistics.stdev(values):.2g}"


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
        lines.append(f"   {run.parent.name}/{run.name}: {finished}, last eval step {last}{stop}; {issues}")
    return lines


def load_metrics(run: Path) -> dict | None:
    path = run / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def section_conditions(runs: list[Path]) -> list[str]:
    lines = ["2. CONDITIONS (mean ± std over seeds; steps / episodes to threshold count from the skill's zero-shot point)"]
    conditions: dict[str, list[dict]] = {}
    for run in runs:
        metrics = load_metrics(run)
        if metrics:
            conditions.setdefault(f"{run.parent.name}/{re.sub(r'_seed\d+$', '', run.name)}", []).append(metrics)
    for condition, metrics_list in sorted(conditions.items()):
        lines.append(f"   {condition}  ({len(metrics_list)} seeds)")
        for mode in ("deterministic", "stochastic"):
            skills = sorted({skill for metrics in metrics_list for skill in metrics["summary"][mode]})
            for skill in skills:
                values = [metrics["summary"][mode].get(skill, {}) for metrics in metrics_list]
                lines.append(
                    f"     {mode[:5]} {skill:10s} steps {mean_std([v.get('steps_to_threshold') for v in values]):>16s}"
                    f"  episodes {mean_std([v.get('episodes_to_threshold') for v in values]):>14s}"
                    f"  auc {mean_std([v.get('auc') for v in values]):>12s}"
                    f"  final {mean_std([v.get('final_success_rate') for v in values]):>12s}"
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


def section_embeddings(runs: list[Path]) -> list[str]:
    lines = ["5. EMBEDDINGS (final learned operator embedding per skill: magnitude (L2 norm); pairwise distance and cosine)"]
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
    from cam.policies.stable_baselines3_policy import ALGORITHMS
    from cam.skills.registry import build_skill

    skills = [build_skill(name, config) for name in config["skills"]]
    encoder = OPERATOR_ENCODERS[config["operator_encoder"]](config, skills)
    inputs = torch.as_tensor(np.stack([encoder.encode(skill.symbolic_action_model) for skill in skills]))
    policy = ALGORITHMS[config["algorithm"]].load(model_path, device="cpu").policy
    extractor = policy.features_extractor if hasattr(policy, "features_extractor") else policy.actor.features_extractor
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
    from cam.policies.stable_baselines3_policy import ALGORITHMS
    from cam.skills.registry import build_skill

    skills = [build_skill(name, config) for name in config["skills"]]
    encoder = OPERATOR_ENCODERS[config["operator_encoder"]](config, skills)
    inputs = torch.as_tensor(np.stack([encoder.encode(skill.symbolic_action_model) for skill in skills]))
    policy = ALGORITHMS[config["algorithm"]].load(model_path, device="cpu").policy
    extractor = policy.features_extractor if hasattr(policy, "features_extractor") else policy.actor.features_extractor
    with torch.no_grad():
        composed = extractor.composed_embedding(inputs)
    return {skill.name: float(magnitude) for skill, magnitude in zip(skills, composed.norm(dim=1))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder of runs, e.g. outputs/overnight_<date>")
    parser.add_argument("--no-embeddings", action="store_true", help="skip section 5 (loads every model)")
    args = parser.parse_args()
    runs = sorted(path.parent for path in args.folder.glob("**/config.json") if (path.parent / "metrics.json").exists())
    if not runs:
        sys.exit(f"no runs with config.json and metrics.json under {args.folder}")
    sections = [section_runs(runs, args.folder), section_conditions(runs), section_curves(runs), section_training(runs)]
    if not args.no_embeddings:
        sections.append(section_embeddings(runs))
    report = "\n\n".join("\n".join(section) for section in sections)
    print(report)
    (args.folder / "review.txt").write_text(report + "\n")
    print(f"\nsaved {args.folder / 'review.txt'}")


if __name__ == "__main__":
    main()
