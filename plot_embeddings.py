"""Plot a 2-D view of the symbolic embeddings and how they move when trained.

Reads the embedding logs written by ``train_multiskill.py`` (one JSON per run)
and projects every embedding vector to 2-D with PCA (a single projection fit
across all runs, so positions are comparable). Then:

* frozen runs are drawn as a single point per skill;
* trainable runs are drawn as a trajectory per skill -- a line from the initial
  (pretrained) embedding to the final learned one, with markers for start/end.

Points/lines are coloured by embedding scheme (mock / name / full) and the
legend is labelled by training mode (scheme + frozen/trainable).

Usage:
    python plot_embeddings.py
    python plot_embeddings.py --log-dir embedding_logs --out embedding_logs/projection.png --show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

SCHEME_COLORS = {
    "mock": "tab:blue",
    "name": "tab:orange",
    "full": "tab:green",
}
_FALLBACK_COLORS = ["tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=Path("embedding_logs"))
    parser.add_argument(
        "--out", type=Path, default=None, help="Output image (default: <log-dir>/embedding_projection.png)."
    )
    parser.add_argument("--show", action="store_true", help="Open an interactive window too.")
    return parser.parse_args()


def load_logs(log_dir: Path) -> list[dict]:
    files = sorted(log_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No embedding logs (*.json) found in {log_dir}")
    logs = []
    for f in files:
        data = json.loads(f.read_text())
        data["_file"] = f.name
        logs.append(data)
    return logs


def _skill_trajectory(log: dict, skill_idx: int) -> np.ndarray:
    """Ordered embedding vectors for one skill: initial then each snapshot."""
    points = [log["initial"][skill_idx]]
    for snap in sorted(log["snapshots"], key=lambda s: s["step"]):
        points.append(snap["embeddings"][skill_idx])
    return np.asarray(points, dtype=float)


def fit_pca_2d(rows: np.ndarray):
    """Return a function projecting (n, dim) arrays to (n, 2) via top-2 PCs."""
    mean = rows.mean(axis=0)
    centered = rows - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    if components.shape[0] < 2:  # degenerate (dim < 2); pad with zeros
        components = np.vstack([components, np.zeros((2 - components.shape[0], rows.shape[1]))])

    def project(matrix: np.ndarray) -> np.ndarray:
        return (np.asarray(matrix, dtype=float) - mean) @ components.T

    return project


def main() -> None:
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logs = load_logs(args.log_dir)

    # Fit one PCA across every vector in every run for comparable positions.
    all_rows = np.vstack(
        [_skill_trajectory(log, i) for log in logs for i in range(len(log["skills"]))]
    )
    project = fit_pca_2d(all_rows)

    fig, ax = plt.subplots(figsize=(10, 8))
    seen_labels: set[str] = set()
    fallback_idx = 0

    for log in logs:
        scheme = log.get("scheme", "?")
        trainable = bool(log.get("trainable", False))
        color = SCHEME_COLORS.get(scheme)
        if color is None:
            color = _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)]
            fallback_idx += 1
        mode = "trainable" if trainable else "frozen"
        label = f"{scheme} ({mode})"
        legend_label = label if label not in seen_labels else None
        seen_labels.add(label)

        for i, skill in enumerate(log["skills"]):
            traj = project(_skill_trajectory(log, i))
            if trainable and len(traj) > 1:
                ax.plot(
                    traj[:, 0], traj[:, 1], "-", color=color, alpha=0.8,
                    label=legend_label,
                )
                legend_label = None  # only label once per run
                ax.scatter(traj[0, 0], traj[0, 1], color=color, marker="o",
                           facecolors="none", s=60, zorder=3)  # start (hollow)
                ax.scatter(traj[-1, 0], traj[-1, 1], color=color, marker="*",
                           s=160, zorder=3)  # end (filled star)
                ax.annotate(
                    skill, traj[-1, :2], textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=color,
                )
            else:
                ax.scatter(traj[0, 0], traj[0, 1], color=color, marker="o", s=70,
                           label=legend_label, zorder=3)
                legend_label = None
                ax.annotate(
                    skill, traj[0, :2], textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=color,
                )

    ax.set_title("Symbolic embeddings (PCA to 2-D)\nhollow o = initial, ★ = learned (trainable)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(title="scheme (mode)", loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = args.out or (args.log_dir / "embedding_projection.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
