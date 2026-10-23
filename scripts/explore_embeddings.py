#!/usr/bin/env python3
"""Explore operator embeddings for RQ5 (do learned embeddings encode the operator's structure?): any embedding
condition, fixed or learned, and the parts it is built from, projected to 2-D or 3-D in an interactive chart, with
tables that compare them on the full vectors. docs/embedding_explorer.md describes what to read in it.

    ./scripts/explore_embeddings.py                                     # the fixed encoders (one-hot, random, multi-hot, text)
    ./scripts/explore_embeddings.py --run outputs/<run> --level operator atom predicate --dims 2 3
    ./scripts/explore_embeddings.py --static multi-hot compositional-tree --level all
    ./scripts/explore_embeddings.py --add 'mine=(:action pickup :parameters (?o - block) :precondition (and ...) ...)'
    ./scripts/explore_embeddings.py --static multi-hot --expression 'raised?=pickup - eff (not (raised ?o)) + eff (raised ?o)'
    ./scripts/explore_embeddings.py --run outputs/<run> --serve         # the same as an app, at http://127.0.0.1:8765

Sources (what is embedded; src/cam/experiments/embedding_points.py):
    --static NAME   an encoder built from main.py's defaults (or --static-config RUN's settings). The operator name is
                    left out of the text and compositional encodings, as in the experiments, unless --operator-name
                    include.
                    compositional-* are untrained: the architecture as a fresh policy has it (--seed)
    --run DIR       a run, as its policy computes it from model.zip and config.json; --extractor picks the features
                    extractor (the actor's by default; with SAC the critic shares it)
A compositional source has untrained twins (--untrained-seeds): the same architecture and settings, freshly
initialized, the control for the structure a composition has by construction; the tables give their mean ± std next
to each number. For a trainable run the control is its input x (level input).

Levels (--level ..., or all; docs/embedding_explorer.md lists them): the operator embedding the policy sees, and
whatever the source is built from: literals, atoms, predicates, the tree's conjunctions, slots, ... Each source and
space is a panel with its own projection (PCA by default, fitted on the registered operators; --method tsne); levels
of a source that share a space are drawn together, and only they are compared.

Custom points are PDDL operators kept in the workspace (--workspace, default outputs/embedding_explorer/workspace.json:
--add LABEL=PDDL or LABEL=@file.pddl, --remove LABEL). Every source embeds them, and each is placed in the fitted
projection without moving the other points. The label is separate from the operator's name, and the name is an input
to one-hot, random, text and compositional-with-name embeddings: keep a registered operator's name to change only its
content. The workspace also keeps --pair A B, --analogy A B C D and --expression LABEL="pickup - eff (not (raised ?o))
+ eff (raised ?o)" (operator labels and literals added up, in sources whose literals share the operators' space).

Output (--out, default outputs/embedding_explorer/<time>/): embeddings.html (self-contained; --plotlyjs cdn for a
small file), points.csv, pairs.csv, vectors.npz, projection.json and tables.txt; the tables are also printed. --serve
runs the same as an app instead, on 127.0.0.1 only (from another machine: ssh -L 8765:127.0.0.1:8765 <machine>).
--offline sends no text to an embedding API: a text missing from the embedding cache is reported instead.
"""

import argparse
import csv
import html
import json
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).with_suffix(".html")
STATIC_SOURCES = ["one-hot", "random", "multi-hot", "text", "compositional-tree", "compositional-slots",
                  "compositional-geometric"]
DEFAULT_STATIC = ["one-hot", "random", "multi-hot", "text"]
# The pairs and analogies the tables answer by default (docs/embedding_explorer.md says why each)
DEFAULT_PAIRS = [["pickup", "pickup-raised"], ["pickup", "unstack"], ["unstack", "unstack-raised"],
                 ["pickup-raised", "unstack-raised"], ["push-left", "push-left-front"], ["putdown", "place-beside"],
                 ["stack", "place-beside"]]
DEFAULT_ANALOGIES = [["pickup", "pickup-raised", "unstack", "unstack-raised"],
                     ["push-left", "push-left-front", "push-right", "push-right-front"],
                     ["push-left", "push-left-back", "push-right", "push-right-back"],
                     ["pickup", "putdown", "unstack", "stack"],
                     ["at-left", "at-right", "at-front", "at-back"]]
# Colour by category: operators by family, components by what their predicate is about. Three hues (the dataviz
# reference palette's first three, the most that stay distinct for every pair of points under colour-vision
# deficiency), each with its own marker shape, so colour is never the only cue.
FAMILIES = {
    "grasp": {"pickup", "pickup-raised", "unstack", "unstack-raised"},
    "place": {"putdown", "stack", "place-beside"},
    "push": {"push-left", "push-right", "push-front", "push-back", "push-left-front", "push-right-front",
             "push-left-back", "push-right-back", "push-beside"},
}
PREDICATE_GROUPS = {
    "region": {"at-left", "at-right", "at-front", "at-back"},
    "relation": {"on", "beside"},
    "object state": {"holding", "on-table", "clear", "raised", "gripper-empty"},
}
CATEGORY_STYLE = {"grasp": (0, "circle"), "place": (1, "square"), "push": (2, "triangle-up"), "region": (0, "circle"),
                  "relation": (1, "square"), "object state": (2, "triangle-up")}
NEUTRAL_STYLE = {"custom": "diamond", "expression": "star", "probe": "x", "component": "circle"}
SYMBOLS_3D = {"triangle-up": "diamond", "diamond": "cross", "star": "x"}  # 3-D markers have fewer shapes
THEMES = {
    "light": {"hues": ["#2a78d6", "#eb6834", "#1baf7a"], "surface": "#fcfcfb", "ink": "#0b0b0b",
              "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"hues": ["#3987e5", "#d95926", "#199e70"], "surface": "#1a1a19", "ink": "#f0efec",
             "secondary": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"},
}
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sources = parser.add_argument_group("sources")
    sources.add_argument("--static", nargs="+", choices=STATIC_SOURCES,
                         help=f"encoders without a run (default {' '.join(DEFAULT_STATIC)} when no --run is given)")
    sources.add_argument("--run", nargs="+", type=Path, default=[], help="run directories (config.json and model.zip)")
    sources.add_argument("--extractor", choices=["actor", "critic", "critic-target", "all"], default="actor",
                         help="which of a run's features extractors (PPO: actor = policy, critic = value function)")
    sources.add_argument("--untrained-seeds", nargs="*", type=int, default=[0, 1, 2],
                         help="seeds of the untrained twins of compositional sources (none: no twins)")
    sources.add_argument("--text-backend", choices=["mock", "openai"], help="static text sources (default: main.py's)")
    sources.add_argument("--operator-text", choices=["pddl", "paragraph"], help="static text sources (default: main.py's)")
    sources.add_argument("--operator-name", choices=["include", "none"],
                         help="static text and compositional sources: the operator name in the encoding or not "
                         "(default none, as in the experiments; with --static-config, the run's)")
    sources.add_argument("--operator-embedding-dim", type=int, help="static sources (default: main.py's)")
    sources.add_argument("--static-config", type=Path, metavar="RUN", help="static sources take this run's settings")
    sources.add_argument("--offline", action="store_true", help="never call a text embedding API: uncached texts are skipped")

    points = parser.add_argument_group("points")
    points.add_argument("--level", nargs="+", default=["operator"], help="levels to plot, or all")
    points.add_argument("--operators", nargs="+", default=["all"],
                        help="registered operators: all, trained (the runs' skills) or names")
    points.add_argument("--all-atoms", action="store_true",
                        help="compositional atoms: every predicate on every ordering of parameter positions")
    points.add_argument("--probe", choices=["literals"],
                        help="add each literal as a one-literal operator (out of distribution for a policy)")
    points.add_argument("--workspace", type=Path, help="default outputs/embedding_explorer/workspace.json")
    points.add_argument("--add", action="append", default=[], metavar="LABEL=PDDL", help="save a custom point (PDDL or @file)")
    points.add_argument("--remove", action="append", default=[], metavar="LABEL", help="remove a custom point")
    points.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"), help="save a pair to compare")
    points.add_argument("--analogy", nargs=4, action="append", default=[], metavar=("A", "B", "C", "D"),
                        help="save an analogy A:B::C:D")
    points.add_argument("--expression", action="append", default=[], metavar="LABEL=EXPRESSION", help="save an expression")

    chart = parser.add_argument_group("chart")
    chart.add_argument("--method", choices=["pca", "tsne"], default="pca")
    chart.add_argument("--dims", nargs="+", type=int, choices=[2, 3], default=[2])
    chart.add_argument("--fit-on", choices=["registered", "trained", "all"], default="registered",
                       help="PCA: the points the projection is fitted on (the others are placed in it)")
    chart.add_argument("--unit-norm", action="store_true", help="scale every vector to length 1 before projecting")
    chart.add_argument("--perplexity", type=float, help="t-SNE (default min(30, (n - 1) / 3))")
    chart.add_argument("--seed", type=int, help="t-SNE, untrained static compositional and random encoders (default 0)")
    chart.add_argument("--out", type=Path, help="output folder (default outputs/embedding_explorer/<time>)")
    chart.add_argument("--plotlyjs", choices=["inline", "cdn"], default="inline")
    chart.add_argument("--no-html", action="store_true", help="tables and data files only")
    chart.add_argument("--serve", action="store_true", help="run as an app on 127.0.0.1 instead of exporting")
    chart.add_argument("--port", type=int, default=8765)
    chart.add_argument("--no-browser", action="store_true")
    return parser.parse_args(argv)


def operator_name_options(include: bool) -> dict:
    """The config keys that put the operator name into the static sources' encodings or leave it out (--operator-name),
    as main.py's --operator-text-name and --compositional-name do."""
    return {"operator_text_name": "include" if include else "none", "compositional_name": "text" if include else "none"}


def with_paths_from_root(config: dict) -> dict:
    """Relative paths in a config (the text embedding cache) are the repository's, wherever this runs from."""
    path = Path(config["text_embedding_cache"])
    return config | {"text_embedding_cache": str(path if path.is_absolute() else ROOT / path)}


def static_config(args: argparse.Namespace) -> dict:
    """The static sources' config: main.py's defaults or --static-config's run, then the options given here."""
    from main import parse_args as main_config

    config = main_config(["--run-directory", str(args.static_config)] if args.static_config else ["--skills", "pickup"])
    overrides = {"text_backend": args.text_backend, "operator_text": args.operator_text, "seed": args.seed,
                 "operator_embedding_dim": args.operator_embedding_dim}
    config |= {key: value for key, value in overrides.items() if value is not None}
    if args.operator_name or not args.static_config:
        config |= operator_name_options(args.operator_name == "include")
    return with_paths_from_root(config)


def untrained_extractor(policy_class, observation_space, action_space, policy_kwargs: dict, seed: int, role: str):
    """The features extractor in role of a policy built afresh with these settings, from seed (the global random state
    is left as it was)."""
    import torch

    from cam.policies.feature_extractors import features_extractors

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        extractors = features_extractors(policy_class(observation_space, action_space, lambda _: 0.0, **policy_kwargs))
    return extractors[role] if role in extractors else extractors["actor"]


def build_sources(args: argparse.Namespace) -> tuple[list, dict[str, list], list[str]]:
    """(sources, untrained twins by source name, notes)."""
    import gymnasium as gym
    from stable_baselines3.sac.policies import MultiInputPolicy

    from cam.experiments.embedding_points import CompositionalSource, FixedEncoderSource, TrainableRunSource, make_offline
    from cam.experiments.environment_setup import compositional_layout
    from cam.policies.feature_extractors import OPERATOR_EMBEDDING_KEY, features_extractors
    from cam.policies.stable_baselines3_policy import ALGORITHMS
    from main import parse_args as main_config, policy_kwargs

    sources, twins, notes = [], {}, []
    static = args.static if args.static is not None else ([] if args.run else DEFAULT_STATIC)
    if static:
        base = static_config(args)
    for name in static:
        if name.startswith("compositional"):
            # untrained: the extractor a new SAC policy starts with (SAC leaves the extractor's initialization as is)
            config = base | {"operator_encoder": "compositional", "compositional_architecture": name.split("-")[1]}
            space = gym.spaces.Dict({"observation": gym.spaces.Box(-1, 1, (1,)), OPERATOR_EMBEDDING_KEY: gym.spaces.Box(
                -np.inf, np.inf, (compositional_layout(config).size,))})
            kwargs = (MultiInputPolicy, space, gym.spaces.Box(-1, 1, (1,)), policy_kwargs(config))
            label = name + ("" if config["compositional_name"] == "text" else " no-name")
            source = CompositionalSource(f"{label} untrained seed {config['seed']}", config,
                                         untrained_extractor(*kwargs, config["seed"], "actor"), all_atoms=args.all_atoms)
            twins[source.name] = [CompositionalSource(f"{label} untrained seed {seed}", config,
                                                      untrained_extractor(*kwargs, seed, "actor"))
                                  for seed in args.untrained_seeds if seed != config["seed"]]
        else:
            config = base | {"operator_encoder": name}
            label = (f"text {config['text_backend']} {config['operator_text']}"
                     + ("" if config["operator_text_name"] == "include" else " no-name") if name == "text" else name)
            source = FixedEncoderSource(label, config)
        sources.append(source)
    for run in args.run:
        config = with_paths_from_root(main_config(["--run-directory", str(run)]))
        model = ALGORITHMS[config["algorithm"]].load(run / "model.zip", device="cpu")
        extractors = features_extractors(model.policy)
        learned = config["operator_encoder"] == "compositional" or config.get("trainable_operator_embedding")
        roles = (list(extractors) if args.extractor == "all" else [args.extractor]) if learned else [None]
        for role in roles:
            if role is not None and role not in extractors:
                notes.append(f"{run.name}: the {role} shares the actor's features extractor, shown as actor")
                role = "actor"
            name = run.name if role is None else f"{run.name} {role}"
            if any(source.name == name for source in sources):
                continue
            if config["operator_encoder"] == "compositional":
                source = CompositionalSource(name, config, extractors[role], config["skills"], args.all_atoms)
                twins[name] = [CompositionalSource(f"{run.name} untrained seed {seed}", config, untrained_extractor(
                    type(model.policy), model.observation_space, model.action_space, model.policy_kwargs, seed, role))
                    for seed in args.untrained_seeds]
            elif learned:
                source = TrainableRunSource(name, config, extractors[role], config["skills"])
            else:
                source = FixedEncoderSource(name, config, config["skills"])
            expected, actual = source.encoder.dim, model.observation_space[OPERATOR_EMBEDDING_KEY].shape[0]
            if expected != actual:
                raise SystemExit(f"{run}: its policy reads operator embeddings of size {actual}, but its config.json "
                                 f"builds {expected} with the current code (a run saved by older code?)")
            sources.append(source)
    if args.offline:
        for source in sources + [twin for group in twins.values() for twin in group]:
            make_offline(source.encoder)
    return sources, twins, notes


# --- workspace: custom points (as PDDL, embedded afresh by every source), pairs, analogies, expressions


def load_workspace(path: Path) -> dict:
    """The saved workspace; a new one starts with the default pairs and analogies."""
    new = {"version": 1, "custom_points": [], "pairs": [list(pair) for pair in DEFAULT_PAIRS],
           "analogies": [list(analogy) for analogy in DEFAULT_ANALOGIES], "expressions": []}
    return new | json.loads(path.read_text()) if path.exists() else new


def save_workspace(path: Path, workspace: dict) -> None:
    from cam.training.checkpointing import atomic_write_text

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(workspace, indent=2) + "\n")


def add_point(workspace: dict, label: str, pddl: str, note: str = "") -> None:
    """Adds a custom point; ValueError (nothing added) for invalid PDDL or a label in use."""
    from cam.experiments.embedding_points import parse_custom
    from cam.skills.registry import SKILL_REGISTRY

    taken = set(SKILL_REGISTRY) | {point["label"] for point in workspace["custom_points"]}
    parse_custom(label, pddl, taken)
    workspace["custom_points"].append({"label": label, "pddl": " ".join(pddl.split()), "note": note,
                                       "added": time.strftime("%Y-%m-%dT%H:%M:%S")})


def remove_point(workspace: dict, label: str) -> None:
    if not any(point["label"] == label for point in workspace["custom_points"]):
        raise ValueError(f"no custom point {label!r}")
    workspace["custom_points"] = [point for point in workspace["custom_points"] if point["label"] != label]


# --- tables: comparisons on the full vectors, within one source and level


@dataclass
class Table:
    title: str
    note: str
    columns: list[str]
    rows: list[list[str]]


def mean_std(values: list[float]) -> str:
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return "n/a"
    return f"{values[0]:.3f}" if len(values) == 1 else f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}"


def number(value: float, digits: int = 3) -> str:
    return "n/a" if value is None or np.isnan(value) else f"{value:.{digits}f}"


def by_source(table: Table) -> tuple[list[str], dict[str, list[list[str]]]]:
    """The table's columns without "source", and its rows grouped by source (run names are long)."""
    index = table.columns.index("source")
    groups: dict[str, list[list[str]]] = {}
    for row in table.rows:
        groups.setdefault(row[index], []).append(row[:index] + row[index + 1 :])
    return table.columns[:index] + table.columns[index + 1 :], groups


def render_text(tables: list[Table]) -> str:
    blocks = []
    for table in tables:
        columns, groups = by_source(table)
        widths = [max(len(str(cell)) for cell in column) for column in zip(columns, *(r for g in groups.values() for r in g))]
        lines = [table.title.upper() + (f" ({table.note})" if table.note else "")]
        lines.append("    " + "  ".join(f"{cell:<{width}}" for cell, width in zip(columns, widths)).rstrip())
        for source, rows in groups.items():
            lines.append(f"  {source}")
            lines += ["    " + "  ".join(f"{cell:<{width}}" for cell, width in zip(row, widths)).rstrip() for row in rows]
        blocks.append("\n".join(lines if groups else lines[:1] + ["  none"]))
    return "\n\n".join(blocks)


def render_html(tables: list[Table]) -> str:
    parts = []
    for table in tables:
        columns, groups = by_source(table)
        head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
        body = "".join(f'<tr><th colspan="{len(columns)}" class="group">{html.escape(source)}</th></tr>' + "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows)
            for source, rows in groups.items()) or f'<tr><td colspan="{len(columns)}">none</td></tr>'
        parts.append(f"<section><h2>{html.escape(table.title)}</h2><p>{html.escape(table.note)}</p>"
                     f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></section>')
    return "\n".join(parts)


REFERENCES = {  # level -> (the structural similarity its agreement is measured against, the sections it counts)
    "pre": ("precondition Jaccard", ("precondition", "negated_precondition")),
    "eff": ("effect Jaccard", ("add", "delete")),
}
LITERALS = ("literal Jaccard", ("precondition", "negated_precondition", "add", "delete"))


def present(groups: list[list[str]], labels) -> list[list[str]]:
    """The pairs or analogies whose labels are all among labels, also as their sections (" pre", " eff": the tree's
    and-sum and and levels have a point per section of each operator)."""
    return [[label + suffix for label in group] for group in groups for suffix in ("", " pre", " eff")
            if all(label + suffix in labels for label in group)]


def operator_level(points: list) -> bool:
    """Whether a level has one point per operator, labelled by it (agreement and pairs between operators)."""
    return bool(points) and all(p.operators == (p.label,) for p in points if p.origin in ("registered", "custom"))


def tables(by_level: dict, twins: dict, workspace: dict, operators: dict, unencodable: list) -> tuple[list[Table], list]:
    """The comparison tables over every source and level (by_level: (source, level) -> points), with each compositional
    source's untrained twins (twins: (source, level) -> [points per twin]), and the pairs table's rows for pairs.csv
    (with each pair's distance relative to the median distance between the registered operators)."""
    from cam.experiments import embedding_geometry as geometry
    from cam.experiments.embedding_points import literal_jaccard, multi_hot_cosine

    def vectors_of(points):
        return {p.label: p.vector.astype(np.float64) for p in points}

    pairs = Table("pairs", "cosine on the full vectors; percentile: share of the registered operators' pairs that are "
                  "less similar; rank: B among A's nearest, then A among B's (ties share their mean rank); untrained: "
                  "the twins' cosine; structure: literal Jaccard and multi-hot cosine",
                  ["A", "B", "source", "level", "cosine", "percentile", "rank", "distance", "untrained",
                   "literal Jaccard", "multi-hot cosine"], [])
    analogies = Table("analogies", "A:B::C:D on the full vectors: cosine of B - A and D - C, |D - C| / |B - A|, and D's "
                      "rank among the nearest to C + B - A (of the level's other points); exact at linear levels "
                      "(multi-hot, W x of multi-hot, the tree's and-sum)",
                      ["analogy", "source", "level", "direction cosine", "length ratio", "rank of D", "untrained"], [])
    agreement = Table("structure agreement", "Spearman between the operators' pairwise cosine and their structural "
                      "similarity (n/a: constant, e.g. one-hot); untrained: the twins' mean ± std",
                      ["source", "level", "against", "Spearman", "untrained"], [])
    between = Table("between levels", "Spearman between two levels' pairwise cosines over the registered operators",
                    ["source", "level A", "level B", "Spearman"], [])
    custom = Table("custom and expression points", "the three nearest registered points by cosine, and the distance "
                   "to the nearest", ["point", "source", "level", "nearest", "distance"], [])
    pair_csv = []
    for (source, level), points in by_level.items():
        vectors = vectors_of(points)
        registered = [p.label for p in points if p.origin in ("registered", "vocabulary")]
        if len(registered) >= 2:
            reference = np.stack([vectors[label] for label in registered])
            similarity = geometry.cosine_matrix(reference)
            upper = np.triu_indices(len(registered), k=1)
            distances = np.linalg.norm(reference[:, None] - reference[None], axis=-1)[upper]
            twin_vectors = [vectors_of(twin) for twin in twins.get((source, level), [])]
            for a, b in present(workspace["pairs"], vectors):
                cosine = geometry.cosine(vectors[a], vectors[b])
                ranks = []
                for first, second in ((a, b), (b, a)):
                    candidates = [label for label in registered if label != first] + ([second] if second not in registered else [])
                    sims = geometry.cosine_matrix(np.stack([vectors[first], *(vectors[c] for c in candidates)]))[0, 1:]
                    ranks.append(geometry.neighbour_rank(sims, candidates.index(second)))
                distance = float(np.linalg.norm(vectors[a] - vectors[b]))
                untrained = mean_std([geometry.cosine(t[a], t[b]) for t in twin_vectors if a in t and b in t]) if twin_vectors else ""
                structure = ([number(literal_jaccard(operators[a], operators[b])),
                              number(multi_hot_cosine(operators[a], operators[b]))]
                             if a in operators and b in operators else ["", ""])
                pairs.rows.append([a, b, source, level, number(cosine), f"{geometry.percentile(cosine, similarity[upper]):.0f}",
                                   f"{ranks[0]:g} / {ranks[1]:g}", f"{distance:.3g}", untrained, *structure])
                pair_csv.append({"A": a, "B": b, "source": source, "level": level, "cosine": cosine,
                                 "percentile": geometry.percentile(cosine, similarity[upper]),
                                 "rank A->B": ranks[0], "rank B->A": ranks[1], "distance": distance,
                                 "relative distance": distance / max(float(np.median(distances)), 1e-12)})
            for a, b, c, d in present(workspace["analogies"], vectors):
                result = geometry.analogy(vectors, a, b, c, d)
                untrained = mean_std([geometry.analogy(t, a, b, c, d)["direction_cosine"] for t in twin_vectors
                                      if all(label in t for label in (a, b, c, d))]) if twin_vectors else ""
                analogies.rows.append([f"{a}:{b}::{c}:{d}", source, level, number(result["direction_cosine"]),
                                       number(result["length_ratio"], 2), f"{result['rank']:g} of {result['candidates']}",
                                       untrained])
            if (level in REFERENCES or points[0].kind == "operator") and operator_level(points) and set(registered) <= set(operators):
                name, sections = REFERENCES.get(level, LITERALS)
                structural = np.array([[literal_jaccard(operators[x], operators[y], sections) for y in registered]
                                       for x in registered])
                twin_values = [geometry.structure_agreement(np.stack([t[label] for label in registered]), structural)
                               for t in twin_vectors if all(label in t for label in registered)]
                agreement.rows.append([source, level, name, number(geometry.structure_agreement(reference, structural)),
                                       mean_std(twin_values) if twin_vectors else ""])
        for point in points:
            if point.origin in ("custom", "expression") and registered:
                candidates = [label for label in registered if label != point.label]
                sims = geometry.cosine_matrix(np.stack([vectors[point.label], *(vectors[c] for c in candidates)]))[0, 1:]
                nearest = np.argsort(-sims, kind="stable")[:3]
                distance = min(np.linalg.norm(vectors[point.label] - vectors[c]) for c in candidates)
                custom.rows.append([point.label, source, level, ", ".join(f"{candidates[i]} {sims[i]:.3f}" for i in nearest),
                                    f"{distance:.3g}"])
    levels_by_source: dict[str, list] = {}
    for (source, level), points in by_level.items():
        if operator_level(points):
            levels_by_source.setdefault(source, []).append((level, vectors_of(points)))
    for source, levels in levels_by_source.items():
        for i, (first, first_vectors) in enumerate(levels):
            for second, second_vectors in levels[i + 1 :]:
                common = [label for label in first_vectors if label in second_vectors and label in operators]
                if len(common) >= 3:
                    value = geometry.spearman(geometry.pair_similarities(np.stack([first_vectors[c] for c in common])),
                                              geometry.pair_similarities(np.stack([second_vectors[c] for c in common])))
                    between.rows.append([source, first, second, number(value)])
    skipped = Table("not encodable", "points a source cannot encode, with its reason", ["source", "level", "label", "reason"],
                    unencodable)
    return [pairs, analogies, agreement, between, custom, skipped], pair_csv


# --- projections and the chart


@dataclass
class Panel:
    source: str
    space: str
    points: list
    coordinates: np.ndarray  # (n, dims)
    residual: np.ndarray | None  # PCA only
    title: str
    projection: dict  # for projection.json


def project_panels(by_level: dict, dims: int, method: str, fit_on: str, unit_norm: bool, seed: int,
                   perplexity: float | None) -> tuple[list[Panel], list[str]]:
    """One panel per source and space, projected to dims (module docstring of embedding_geometry)."""
    from cam.experiments import embedding_geometry as geometry

    groups: dict[tuple[str, str], list] = {}
    for (source, _), points in by_level.items():
        for point in points:
            groups.setdefault((source, point.space), []).append(point)
    panels, warnings = [], []
    for (source, space), points in groups.items():
        vectors = np.stack([p.vector.astype(np.float64) for p in points])
        if unit_norm:
            vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        if method == "tsne":
            coordinates, notes = geometry.fit_tsne(vectors, dims, perplexity, seed=seed)
            warnings += [f"{source} · {space}: {note}" for note in notes]
            panels.append(Panel(source, space, points, coordinates, None, f"{html.escape(source)}<br>{space} · t-SNE",
                                {"source": source, "space": space, "method": "tsne", "labels": [p.label for p in points]}))
            continue
        fitted = {"registered": [p.origin in ("registered", "vocabulary") for p in points],
                  "trained": [bool(p.trained) for p in points], "all": [True] * len(points)}[fit_on]
        if not any(fitted):
            warnings.append(f"{source} · {space}: no {fit_on} points to fit the projection on; fitted on all")
            fitted = [True] * len(points)
        projection = geometry.fit_pca(vectors[np.array(fitted)], dims)
        coordinates, residual = geometry.project(projection, vectors)
        explained = ", ".join(f"PC{k + 1} {value:.0%}" for k, value in enumerate(projection.explained))
        panels.append(Panel(source, space, points, coordinates, residual, f"{html.escape(source)}<br>{space} · {explained}", {
            "source": source, "space": space, "method": "pca", "fit_on": [p.label for p, f in zip(points, fitted) if f],
            "explained": projection.explained.tolist(), "mean": projection.mean.tolist(),
            "components": projection.components.tolist()}))
    return panels, warnings


def category(point) -> str:
    if point.origin in ("custom", "expression", "probe"):
        return point.origin
    if point.kind in ("literal", "atom", "feature", "predicate"):
        words = set(re.findall(r"[a-z][a-z-]*", point.label))
        return next((group for group, predicates in PREDICATE_GROUPS.items() if words & predicates), "component")
    if point.kind in ("type", "variable", "typed-variable"):
        return "component"
    return next((family for family, members in FAMILIES.items() if point.operators and point.operators[0] in members),
                "component")


def hover(point, residual: float | None) -> str:
    lines = [f"<b>{html.escape(point.label)}</b>", f"{html.escape(point.level)} · {html.escape(point.source)}",
             point.origin + ("" if point.trained is None else (" · trained" if point.trained else " · held out"))]
    if point.operators and point.operators != (point.label,):
        lines.append("in " + html.escape(", ".join(point.operators[:8])) + (" ..." if len(point.operators) > 8 else ""))
    lines.append(f"norm {np.linalg.norm(point.vector):.3g}" + ("" if residual is None else f" · off the plane {residual:.0%}"))
    if point.text:
        lines += [html.escape(line) for line in point.text.splitlines()]
    return "<br>".join(lines)


def figure(panels: list[Panel], dims: int, theme: str, labels: bool, pairs: list, analogies: list) -> dict:
    """The plotly figure (a JSON-ready dict) of the panels, as small multiples two to a row. Pairs, analogies and a
    trainable run's x -> W x are lines and arrows, hidden until their legend entry is clicked."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    colors, axes = THEMES[theme], "xyz"[:dims]
    columns = 1 if len(panels) == 1 else 2
    rows = max(1, -(-len(panels) // columns))
    fig = make_subplots(rows=rows, cols=columns, specs=[[{"type": "scene" if dims == 3 else "xy"}] * columns] * rows,
                        subplot_titles=[panel.title for panel in panels] or [""], horizontal_spacing=0.08,
                        vertical_spacing=0.2 / rows)
    shown = set()

    def add(trace: dict, index: int) -> None:
        trace["showlegend"] = trace["name"] not in shown  # one legend entry for every panel's trace of a name
        shown.add(trace["name"])
        fig.add_trace((go.Scatter3d if dims == 3 else go.Scatter)(**trace), row=index // columns + 1, col=index % columns + 1)

    for index, panel in enumerate(panels):
        groups: dict[tuple, list[int]] = {}
        for i, point in enumerate(panel.points):
            groups.setdefault((category(point), point.trained is False), []).append(i)
        for (name, held_out), members in groups.items():
            hue, symbol = CATEGORY_STYLE.get(name, (None, NEUTRAL_STYLE.get(name, "circle")))
            color = colors["secondary"] if hue is None else colors["hues"][hue]
            symbol = (SYMBOLS_3D.get(symbol, symbol) if dims == 3 else symbol) + ("-open" if held_out else "")
            legend = name + (" (held out)" if held_out else "")
            add({**{axis: panel.coordinates[members, k] for k, axis in enumerate(axes)},
                 "mode": "markers+text" if labels else "markers", "name": legend, "legendgroup": legend,
                 "marker": {"color": color, "symbol": symbol, "size": 5 if dims == 3 else (9 if hue is not None and "triangle" not in symbol else 11),
                            "line": {"color": color if held_out else colors["surface"], "width": 2}},
                 "text": [panel.points[i].label for i in members], "textposition": "top center",
                 "textfont": {"color": colors["secondary"], "size": 10}, "hoverinfo": "text",
                 "hovertext": [hover(panel.points[i], None if panel.residual is None else panel.residual[i]) for i in members]},
                index)
        position = {(point.level, point.label): i for i, point in enumerate(panel.points)}
        links: dict[str, list] = {"pair": [], "analogy": []}
        for level in dict.fromkeys(point.level for point in panel.points):
            labels = {label for at, label in position if at == level}
            links["pair"] += [((level, a), (level, b)) for a, b in present(pairs, labels)]
            links["analogy"] += [((level, x), (level, y)) for a, b, c, d in present(analogies, labels) for x, y in ((a, b), (c, d))]
        links["x -> W x"] = [(("input", p.label), ("operator", p.label)) for p in panel.points
                             if p.level == "operator" and ("input", p.label) in position]
        for kind, segments in links.items():
            if not segments:
                continue
            path = {axis: [value for start, end in segments for value in (panel.coordinates[position[start], k],
                                                                           panel.coordinates[position[end], k], None)]
                    for k, axis in enumerate(axes)}
            trace = {**path, "mode": "lines", "name": kind, "legendgroup": kind, "visible": "legendonly",
                     "hoverinfo": "skip", "line": {"color": colors["muted"], "width": 1 if kind == "pair" else 1.5}}
            if dims == 2 and kind != "pair":  # an arrowhead at the end of each segment
                trace |= {"mode": "lines+markers", "marker": {"symbol": "arrow", "angleref": "previous",
                                                              "size": [0, 10, 0] * len(segments), "color": colors["muted"]}}
            add(trace, index)
    style = {"gridcolor": colors["grid"], "zerolinecolor": colors["axis"], "linecolor": colors["axis"], "showline": True,
             "tickfont": {"color": colors["muted"]}}
    height = 480 * rows
    fig.update_layout(template="none", height=height, paper_bgcolor=colors["surface"], plot_bgcolor=colors["surface"],
                      font={"family": FONT, "color": colors["secondary"], "size": 12},
                      margin={"l": 40, "r": 20, "t": 90, "b": 80}, hoverlabel={"font": {"family": FONT}},
                      legend={"orientation": "h", "x": 0, "y": -50 / height, "yanchor": "top"})
    fig.update_annotations(font={"color": colors["ink"], "size": 13})  # the panel titles
    if dims == 3:
        names = ["PC1", "PC2", "PC3"] if panels and panels[0].residual is not None else ["t-SNE 1", "t-SNE 2", "t-SNE 3"]
        fig.update_scenes(aspectmode="data", **{f"{axis}axis": style | {"showbackground": False, "title": {"text": name}}
                                                for axis, name in zip(axes, names)})
    else:
        fig.update_xaxes(**style)
        fig.update_yaxes(**style)
        for index in range(len(panels)):
            suffix = "" if index == 0 else str(index + 1)
            fig.update_layout({f"yaxis{suffix}": {"scaleanchor": f"x{suffix}", "scaleratio": 1}})  # equal aspect
    return json.loads(fig.to_json())


# --- the explorer: sources, points and tables for one request; the export and the app share it


class Explorer:
    def __init__(self, args: argparse.Namespace):
        from cam.experiments.embedding_points import registered_operators

        self.args = args
        self.workspace_path = args.workspace or ROOT / "outputs" / "embedding_explorer" / "workspace.json"
        self.workspace = load_workspace(self.workspace_path)
        self.sources, self.twins, self.notes = build_sources(args)
        every = registered_operators()
        trained = {name for source in self.sources for name in (source.trained_skills or ())}
        names = {"all": {entry.label for entry in every}, "trained": trained}.get(" ".join(args.operators), set(args.operators))
        unknown = names - {entry.label for entry in every}
        if unknown or not names:
            raise SystemExit(f"--operators: {sorted(unknown) or 'trained needs a --run'} (registered: {[e.label for e in every]})")
        self.registered = [entry for entry in every if entry.label in names]

    def defaults(self) -> dict:
        return {"sources": [source.name for source in self.sources], "levels": self.args.level, "method": self.args.method,
                "dims": self.args.dims, "fit_on": self.args.fit_on, "unit_norm": self.args.unit_norm, "labels": True}

    def operators(self) -> tuple[list, list[list[str]]]:
        """The registered and custom operators (and --probe literals' one-literal operators), and problems."""
        from cam.domain.pddl.pddl_parser import parse_operator
        from cam.experiments.embedding_points import LabelledOperator, literal_components

        entries, problems = list(self.registered), []
        for point in self.workspace["custom_points"]:
            try:
                entries.append(LabelledOperator(point["label"], parse_operator(point["pddl"]), "custom"))
            except ValueError as error:
                problems.append(["workspace", "", point["label"], str(error)])
        if self.args.probe == "literals":
            entries += [LabelledOperator(label, operator, "probe")
                        for label, operator, _ in literal_components(self.registered).values()]
        return entries, problems

    def expression_points(self, source, operators: list) -> tuple[list, list[list[str]]]:
        """Each workspace expression, evaluated in a source whose literals share its operators' space."""
        from cam.experiments.embedding_points import FixedEncoderSource, LabelledOperator, parse_expression

        if not self.workspace["expressions"]:
            return [], []
        if not isinstance(source, FixedEncoderSource) or "literal" not in {level.name for level in source.levels()}:
            return [], [[source.name, "operator", e["label"], "its literals and operators are in different spaces; add "
                         "the edited operator as a custom point instead"] for e in self.workspace["expressions"]]
        points, problems, by_label = [], [], {entry.label: entry.operator for entry in operators}
        for expression in self.workspace["expressions"]:
            try:
                terms = parse_expression(expression["expression"], by_label)
                _, inputs, unencodable = source.encode([LabelledOperator(str(i), op) for i, (_, _, op) in enumerate(terms)])
                if unencodable:
                    raise ValueError("; ".join(unencodable.values()))
            except ValueError as error:
                problems.append([source.name, "operator", expression["label"], str(error)])
                continue
            vector = sum(sign * row for (sign, _, _), row in zip(terms, source.output("operator", inputs)))
            members = [term for _, term, _ in terms if term in by_label]
            points.append(source.point(source.level("operator"), expression["label"], vector, "expression", members,
                                       expression["expression"]))
        return points, problems

    def collect(self, source_names: list[str], levels: list[str]) -> tuple[dict, dict, list[list[str]]]:
        """(points by (source, level), the untrained twins' points by (source, level), what could not be encoded)."""
        operators, problems = self.operators()
        by_level, twins = {}, {}
        for source in self.sources:
            if source.name not in source_names:
                continue
            names = [level.name for level in source.levels()]
            for level in (names if "all" in levels else [level for level in levels if level in names]):
                points, unencodable = source.points(level, operators)
                if level == "operator":
                    extra, refused = self.expression_points(source, operators)
                    points, problems = points + extra, problems + refused
                problems += [[source.name, level, label, reason] for label, reason in unencodable.items()]
                by_level[(source.name, level)] = points
                twins[(source.name, level)] = [twin.points(level, self.registered)[0] for twin in self.twins.get(source.name, [])]
        return by_level, twins, problems

    def chart(self, request: dict) -> dict:
        """{figures: {dims: {theme: figure}}, tables_html, tables_text, warnings} for a chart request (defaults())."""
        request = self.defaults() | request
        if request["method"] not in ("pca", "tsne") or request["fit_on"] not in ("registered", "trained", "all"):
            raise ValueError(f"unknown method or fit_on in {request}")
        by_level, twins, problems = self.collect(request["sources"], request["levels"])
        operators = {entry.label: entry.operator for entry in self.operators()[0]}
        result_tables, pair_rows = tables(by_level, twins, self.workspace, operators, problems)
        figures, warnings, panels = {}, list(self.notes), {}
        for dims in request["dims"]:
            drawn, notes = project_panels(by_level, int(dims), request["method"], request["fit_on"],
                                          request["unit_norm"], self.args.seed or 0, self.args.perplexity)
            warnings += notes
            figures[str(dims)] = {theme: figure(drawn, int(dims), theme, request["labels"], self.workspace["pairs"],
                                                self.workspace["analogies"]) for theme in THEMES}
            panels[str(dims)] = drawn
        if not by_level:
            warnings.append("no points: no chosen source has the chosen levels")
        return {"figures": figures, "tables_html": render_html(result_tables), "tables_text": render_text(result_tables),
                "warnings": warnings, "by_level": by_level, "panels": panels, "tables": result_tables,
                "pair_rows": pair_rows}

    def export(self, request: dict, out: Path, plotlyjs: str = "inline", write_html: bool = True) -> dict:
        """Writes the chart and its data to out (module docstring); returns the chart."""
        result = self.chart(request)
        out.mkdir(parents=True, exist_ok=True)
        (out / "tables.txt").write_text(result["tables_text"] + "\n")
        (out / "projection.json").write_text(json.dumps(
            {dims: [panel.projection for panel in drawn] for dims, drawn in result["panels"].items()}, indent=1))
        arrays = {}
        for (source, level), points in result["by_level"].items():
            if points:
                key = f"{source} | {level}".replace("/", "_")
                arrays[key] = np.stack([p.vector for p in points])
                arrays[key + " | labels"] = np.array([p.label for p in points])
        np.savez(out / "vectors.npz", **arrays)
        coordinates = {dims: {(p.source, p.level, p.label): (*panel.coordinates[i], "" if panel.residual is None
                                                             else panel.residual[i])
                              for panel in drawn for i, p in enumerate(panel.points)}
                       for dims, drawn in result["panels"].items()}
        with open(out / "points.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["source", "level", "space", "label", "kind", "origin", "trained", "operators", "norm"]
                            + [f"{dims}-D {axis}" for dims in coordinates for axis in [*"xyz"[: int(dims)], "residual"]])
            for points in result["by_level"].values():
                for p in points:
                    writer.writerow([p.source, p.level, p.space, p.label, p.kind, p.origin, p.trained,
                                     " ".join(p.operators), f"{np.linalg.norm(p.vector):.6g}",
                                     *(value for at in coordinates.values() for value in at[(p.source, p.level, p.label)])])
        pair_rows = result["pair_rows"]
        with open(out / "pairs.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]) if pair_rows else ["A", "B"])
            writer.writeheader()
            writer.writerows(pair_rows)
        if write_html:
            (out / "embeddings.html").write_text(page(plotlyjs, {key: result[key] for key in ("figures", "tables_html", "warnings")}))
        return result

    # The app's JSON API (serve() routes requests here): each returns (HTTP status, JSON-ready body).

    def api_options(self) -> tuple[int, dict]:
        from cam.domain.pddl.pddl_writer import to_pddl

        return 200, {
            "sources": [{"name": source.name, "levels": [vars(level) for level in source.levels()]} for source in self.sources],
            "operators": [{"label": entry.label, "pddl": to_pddl(entry.operator)} for entry in self.registered],
            "workspace": self.workspace, "defaults": self.defaults(),
        }

    def api_chart(self, body: dict) -> tuple[int, dict]:
        result = self.chart(body)
        return 200, {key: result[key] for key in ("figures", "tables_html", "warnings")}

    def api_add_point(self, body: dict) -> tuple[int, dict]:
        """Adds and saves a custom point, and says which sources cannot encode it."""
        from cam.domain.pddl.pddl_parser import parse_operator
        from cam.experiments.embedding_points import LabelledOperator

        label, pddl = body.get("label", "").strip(), body.get("pddl", "")
        add_point(self.workspace, label, pddl, body.get("note", ""))
        save_workspace(self.workspace_path, self.workspace)
        entry = LabelledOperator(label, parse_operator(pddl), "custom")
        errors = {source.name: source.points("operator", [entry])[1].get(entry.label) for source in self.sources}
        return 200, {"label": entry.label, "errors": {name: error for name, error in errors.items() if error}}

    def api_remove_point(self, label: str) -> tuple[int, dict]:
        remove_point(self.workspace, label)
        save_workspace(self.workspace_path, self.workspace)
        return 200, {"removed": label}

    def api_list(self, key: str, body: dict, width: int) -> tuple[int, dict]:
        """Replaces the workspace's pairs (width 2) or analogies (width 4)."""
        items = body.get(key, [])
        if not all(isinstance(item, list) and len(item) == width and all(isinstance(x, str) for x in item) for item in items):
            raise ValueError(f"{key}: a list of {width} labels each")
        self.workspace[key] = items
        save_workspace(self.workspace_path, self.workspace)
        return 200, {key: items}

    def api_export(self, body: dict) -> tuple[int, dict]:
        out = self.args.out or ROOT / "outputs" / "embedding_explorer" / time.strftime("%Y%m%d-%H%M%S")
        self.export(body, out, self.args.plotlyjs)
        return 200, {"folder": str(out)}


def page(plotlyjs: str, data: dict | None) -> str:
    """The app page (scripts/explore_embeddings.html); with data, the self-contained export of one chart."""
    from plotly.offline import get_plotlyjs, get_plotlyjs_version

    script = {"inline": lambda: f"<script>{get_plotlyjs()}</script>",
              "cdn": lambda: f'<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"></script>',
              "served": lambda: '<script src="/plotly.js"></script>'}[plotlyjs]()
    payload = json.dumps(data).replace("</", "<\\/") if data is not None else "null"
    return PAGE.read_text().replace("<!--PLOTLY-->", script).replace("/*DATA*/null", payload)


def make_server(explorer: Explorer, port: int):
    """The app's server on 127.0.0.1:port (0: any free port): the page, plotly.js from the installed package, and
    Explorer's JSON API."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import unquote

    from plotly.offline import get_plotlyjs

    class Handler(BaseHTTPRequestHandler):
        def send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def respond(self, call) -> None:
            try:
                status, body = call()
            except ValueError as error:  # a request the explorer refuses, e.g. invalid PDDL
                status, body = 400, {"error": str(error)}
            except Exception as error:  # shown on the page, and the traceback here
                traceback.print_exc()
                status, body = 500, {"error": f"{type(error).__name__}: {error}"}
            self.send(status, json.dumps(body).encode(), "application/json")

        def body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self) -> None:
            if self.path == "/":
                self.send(200, page("served", None).encode(), "text/html; charset=utf-8")
            elif self.path == "/plotly.js":
                self.send(200, get_plotlyjs().encode(), "application/javascript")
            elif self.path == "/api/options":
                self.respond(explorer.api_options)
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            routes = {"/api/chart": explorer.api_chart, "/api/points": explorer.api_add_point,
                      "/api/pairs": lambda body: explorer.api_list("pairs", body, 2),
                      "/api/analogies": lambda body: explorer.api_list("analogies", body, 4),
                      "/api/export": explorer.api_export}
            if self.path not in routes:
                return self.send(404, b"not found", "text/plain")
            body = self.body()
            self.respond(lambda: routes[self.path](body))

        def do_DELETE(self) -> None:
            if not self.path.startswith("/api/points/"):
                return self.send(404, b"not found", "text/plain")
            self.respond(lambda: explorer.api_remove_point(unquote(self.path.removeprefix("/api/points/"))))

        def log_message(self, format, *arguments) -> None:
            pass

    return HTTPServer(("127.0.0.1", port), Handler)


def serve(explorer: Explorer, port: int, open_browser: bool) -> None:
    import webbrowser

    server = make_server(explorer, port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"serving the embedding explorer at {url} (Ctrl+C stops it)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def update_workspace(explorer: Explorer, args: argparse.Namespace) -> None:
    """--add, --remove, --pair, --analogy, --expression: saved to the workspace before anything is computed."""
    workspace, changed = explorer.workspace, False
    for item in args.add:
        label, _, pddl = item.partition("=")
        pddl = Path(pddl[1:]).read_text() if pddl.startswith("@") else pddl
        try:
            add_point(workspace, label.strip(), pddl)
        except ValueError as error:
            raise SystemExit(f"--add {label}: {error}")
        changed = True
    for label in args.remove:
        try:
            remove_point(workspace, label)
        except ValueError as error:
            raise SystemExit(f"--remove: {error}")
        changed = True
    for key, items in (("pairs", args.pair), ("analogies", args.analogy)):
        new = [list(item) for item in items if list(item) not in workspace[key]]
        workspace[key] += new
        changed |= bool(new)
    for item in args.expression:
        label, _, expression = item.partition("=")
        workspace["expressions"] = [e for e in workspace["expressions"] if e["label"] != label.strip()]
        workspace["expressions"].append({"label": label.strip(), "expression": expression.strip()})
        changed = True
    if changed:
        save_workspace(explorer.workspace_path, workspace)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sys.path.insert(0, str(ROOT))
    explorer = Explorer(args)
    update_workspace(explorer, args)
    if args.serve:
        serve(explorer, args.port, not args.no_browser)
        return
    out = args.out or ROOT / "outputs" / "embedding_explorer" / time.strftime("%Y%m%d-%H%M%S")
    result = explorer.export({}, out, args.plotlyjs, not args.no_html)
    for warning in result["warnings"]:
        print(f"note: {warning}")
    print(result["tables_text"])
    print(f"\nsaved {out}" + ("" if args.no_html else f" (open {out / 'embeddings.html'})"))


if __name__ == "__main__":
    main()
