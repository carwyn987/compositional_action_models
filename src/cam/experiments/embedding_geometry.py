"""Geometry of operator embeddings for RQ5 (scripts/explore_embeddings.py): projections to 2 or 3 dimensions for
plotting, and the comparisons that do not depend on a projection. Vectors are full-dimensional numpy arrays; nothing
here knows what they embed.

Projections. PCA is fitted on a reference set (by default the registered operators) and every other point (custom,
probe, expression) is placed with the same mean and components, so adding a point never moves the others. A point
off the fitted plane is drawn where its projection falls, so each one also gets its residual: the fraction of its
distance from the mean that the plane does not show (0 on the plane, 1 orthogonal to it). Each component's sign is
fixed so that the reference point farthest along it is positive, which keeps the picture from mirroring between
runs. t-SNE (scikit-learn) keeps neighbourhoods rather than distances, needs a few dozen points, and is refitted on
every point: positions, and distances between clusters, mean nothing on their own.

Comparisons, always within one space (embedding_points.Level.space) and on the full vectors:
    cosine          cosine similarity
    percentile      the share of reference pairs that are less similar than this pair
    neighbour rank  where B is among A's neighbours by cosine (1: nearest; tied neighbours share their mean rank)
    analogy         A:B::C:D, e.g. pickup:pickup-raised::unstack:unstack-raised: the cosine between the directions
                    B - A and D - C, their length ratio, and D's rank among the neighbours of C + B - A
    agreement       Spearman correlation between the embeddings' pairwise cosine and a structural similarity (e.g.
                    the operators' literal Jaccard): representational similarity analysis. nan when either side is
                    constant (one-hot: every pair has cosine 0)
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class Projection:
    method: str  # pca | tsne
    mean: np.ndarray  # pca: the reference set's mean
    components: np.ndarray  # pca: (dims, dim) orthonormal rows
    explained: np.ndarray  # pca: the fraction of the reference set's variance along each component


def fit_pca(reference: np.ndarray, dims: int) -> Projection:
    """PCA of the reference vectors (rows), with the sign convention of the module docstring. A reference set with
    fewer directions than dims gets zero components for the rest."""
    mean = reference.mean(axis=0)
    centered = reference - mean
    _, singular, rows = np.linalg.svd(centered, full_matrices=False)
    components = np.zeros((dims, reference.shape[1]))
    count = min(dims, len(rows))
    components[:count] = rows[:count]
    coordinates = centered @ components.T
    for k in range(dims):
        farthest = np.argmax(np.abs(coordinates[:, k]))
        if coordinates[farthest, k] < 0:
            components[k] *= -1
    variance = singular**2
    explained = np.zeros(dims)
    if variance.sum() > 0:
        explained[:count] = variance[:count] / variance.sum()
    return Projection("pca", mean, components, explained)


def project(projection: Projection, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, dims) coordinates of the vectors in the fitted PCA, and each one's residual (module docstring)."""
    centered = vectors - projection.mean
    coordinates = centered @ projection.components.T
    off_plane = np.linalg.norm(centered - coordinates @ projection.components, axis=1)
    distance = np.linalg.norm(centered, axis=1)
    return coordinates, np.divide(off_plane, distance, out=np.zeros_like(distance), where=distance > 1e-12)


def fit_tsne(vectors: np.ndarray, dims: int, perplexity: float | None = None, metric: str = "euclidean",
             seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """(n, dims) t-SNE coordinates and warnings. The perplexity defaults to min(30, (n - 1) / 3), and must be below
    n; below 5 points this falls back to PCA, and below 20 it warns that the layout is unreliable."""
    count = len(vectors)
    if count < 5:
        coordinates, _ = project(fit_pca(vectors, dims), vectors)
        return coordinates, [f"t-SNE needs at least 5 points ({count} here): PCA shown instead"]
    warnings = ["refit: every position may change; distances between clusters are not meaningful"]
    if count < 20:
        warnings.append(f"t-SNE with {count} points: the layout is unreliable")
    perplexity = min(30.0, (count - 1) / 3) if perplexity is None else perplexity
    if not 0 < perplexity < count:
        raise ValueError(f"perplexity must be between 0 and the number of points ({count}); got {perplexity}")
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=dims, perplexity=perplexity, init="pca", metric=metric, random_state=seed)
    return tsne.fit_transform(np.asarray(vectors, dtype=np.float64)), warnings


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = np.divide(vectors, norms, out=np.zeros_like(vectors, dtype=np.float64), where=norms > 1e-12)
    return unit @ unit.T


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_matrix(np.stack([a, b]))[0, 1])


def percentile(value: float, reference: np.ndarray) -> float:
    """The share (0-100) of the reference values below value, ties counting half."""
    return 100.0 * (np.sum(reference < value) + 0.5 * np.sum(reference == value)) / len(reference)


def neighbour_rank(similarities: np.ndarray, candidate: int) -> float:
    """candidate's rank (1: most similar) among the given similarities to one point; tied candidates share their
    mean rank, so a point that is equally similar to every other gets the middle rank."""
    value = similarities[candidate]
    return 1.0 + np.sum(similarities > value) + 0.5 * (np.sum(similarities == value) - 1)


def pair_similarities(vectors: np.ndarray) -> np.ndarray:
    """The cosine of every distinct pair of rows (the upper triangle)."""
    return cosine_matrix(vectors)[np.triu_indices(len(vectors), k=1)]


def analogy(vectors: dict[str, np.ndarray], a: str, b: str, c: str, d: str) -> dict[str, float]:
    """A:B::C:D (module docstring): direction cosine, length ratio |D - C| / |B - A|, and the rank of D among the
    neighbours of C + B - A, every vector but A, B and C being a candidate. Direction and ratio are nan when A and B,
    or C and D, are the same point up to float error (e.g. the precondition embeddings of pickup and pickup-raised)."""
    first, second = vectors[b] - vectors[a], vectors[d] - vectors[c]
    candidates = [label for label in vectors if label not in (a, b, c)]
    target = vectors[c] + first
    similarities = cosine_matrix(np.stack([target, *(vectors[label] for label in candidates)]))[0, 1:]
    scale = max(np.linalg.norm(vectors[label]) for label in (a, b, c, d))
    degenerate = min(np.linalg.norm(first), np.linalg.norm(second)) <= 1e-5 * scale
    return {
        "direction_cosine": float("nan") if degenerate else cosine(first, second),
        "length_ratio": float("nan") if degenerate else float(np.linalg.norm(second) / np.linalg.norm(first)),
        "rank": neighbour_rank(similarities, candidates.index(d)),
        "candidates": len(candidates),
    }


def ranks(values: np.ndarray) -> np.ndarray:
    """Ranks from 1, tied values sharing their mean rank."""
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values))
    result[order] = np.arange(1, len(values) + 1)
    for value in np.unique(values):
        tied = values == value
        result[tied] = result[tied].mean()
    return result


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rank correlation; nan when either side is constant."""
    rx, ry = ranks(np.asarray(x, dtype=np.float64)), ranks(np.asarray(y, dtype=np.float64))
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def structure_agreement(vectors: np.ndarray, structural_similarity: np.ndarray) -> float:
    """Spearman between the rows' pairwise cosine and an (n, n) structural similarity, over distinct pairs."""
    upper = np.triu_indices(len(vectors), k=1)
    return spearman(cosine_matrix(vectors)[upper], structural_similarity[upper])
