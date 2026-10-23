"""embedding_geometry: projections that never move the reference points, and comparisons with known answers."""

import numpy as np
import pytest

from cam.experiments import embedding_geometry as geometry


@pytest.mark.unit
def test_points_added_to_a_pca_leave_the_reference_coordinates_unchanged():
    """The projection is fitted on the reference set only, so projecting more points (here a duplicate of a reference
    point and a point off the plane) changes no reference coordinate; the duplicate lands on its original with
    residual 0, and the other point reports what the plane hides. The control: refitting on every point moves the
    reference points."""
    rng = np.random.default_rng(0)
    reference = rng.standard_normal((6, 10))
    projection = geometry.fit_pca(reference, 2)
    before, residual = geometry.project(projection, reference)
    off_plane = reference.mean(axis=0) + 5 * np.linalg.svd(reference - reference.mean(axis=0))[2][4]
    after, added_residual = geometry.project(projection, np.vstack([reference, reference[2], off_plane]))
    np.testing.assert_array_equal(after[:6], before)
    np.testing.assert_allclose(after[6], before[2])
    assert added_residual[6] == pytest.approx(residual[2]) and added_residual[7] == pytest.approx(1.0)
    refitted, _ = geometry.project(geometry.fit_pca(np.vstack([reference, off_plane]), 2), reference)
    assert not np.allclose(refitted, before)


@pytest.mark.unit
def test_pca_of_one_hot_vectors_explains_equal_shares_and_has_a_fixed_sign():
    """n one-hot vectors are equidistant: every principal direction explains 1 / (n - 1) of the variance. Each
    component points to the reference point farthest along it, so negating the data does not mirror the picture."""
    vectors = np.eye(5)
    projection = geometry.fit_pca(vectors, 3)
    np.testing.assert_allclose(projection.explained, [0.25, 0.25, 0.25])
    for data in (vectors, -vectors):
        coordinates, _ = geometry.project(geometry.fit_pca(data, 3), data)
        assert all(coordinates[np.argmax(np.abs(coordinates[:, k])), k] > 0 for k in range(3))


@pytest.mark.unit
def test_tsne_perplexity_fallback_and_determinism():
    """The default perplexity is below the number of points; below 5 points t-SNE falls back to PCA (with a warning);
    the same seed gives the same layout. Skipped without scikit-learn, which only the explorer's t-SNE needs, so a
    machine set up for training alone still passes the unit tests start_overnight.sh runs first."""
    pytest.importorskip("sklearn")
    vectors = np.random.default_rng(1).standard_normal((12, 8))
    first, warnings = geometry.fit_tsne(vectors, 2, seed=3)
    second, _ = geometry.fit_tsne(vectors, 2, seed=3)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (12, 2) and any("unreliable" in warning for warning in warnings)
    with pytest.raises(ValueError):
        geometry.fit_tsne(vectors, 2, perplexity=12)
    few, warnings = geometry.fit_tsne(vectors[:4], 2)
    np.testing.assert_allclose(few, geometry.project(geometry.fit_pca(vectors[:4], 2), vectors[:4])[0])
    assert "PCA shown instead" in warnings[0]


@pytest.mark.unit
def test_neighbour_rank_and_percentile_share_ties():
    """Ties share their mean rank: a point equally similar to 4 others ranks each 2.5th, not 1st (the one-hot case)."""
    assert geometry.neighbour_rank(np.zeros(4), 1) == 2.5
    assert geometry.neighbour_rank(np.array([0.9, 0.1, 0.5]), 2) == 2
    assert geometry.percentile(0.5, np.array([0.1, 0.5, 0.9, 1.0])) == 37.5


@pytest.mark.unit
def test_an_exact_analogy_and_its_control():
    """D - C equal to B - A gives direction cosine 1, length ratio 1 and D nearest to C + B - A; moving D off the
    parallelogram lowers all three."""
    vectors = {"a": np.array([0.0, 0, 0]), "b": np.array([1.0, 0, 0]), "c": np.array([0.0, 1, 0]),
               "d": np.array([1.0, 1, 0]), "e": np.array([1.0, 0.9, 0])}
    exact = geometry.analogy(vectors, "a", "b", "c", "d")
    assert exact["direction_cosine"] == pytest.approx(1) and exact["length_ratio"] == pytest.approx(1)
    assert exact["rank"] == 1
    off = geometry.analogy(vectors | {"d": np.array([0.0, 1, 1])}, "a", "b", "c", "d")
    assert off["direction_cosine"] < 0.5 and off["rank"] == 2


@pytest.mark.unit
def test_structure_agreement_is_nan_for_constant_similarities():
    """Spearman of pairwise cosines against a structural similarity: 1 when they agree in order, nan when the
    embeddings carry no order (one-hot: every cosine 0)."""
    structural = np.array([[1, 0.8, 0.1], [0.8, 1, 0.3], [0.1, 0.3, 1]])
    vectors = np.array([[1.0, 0.0], [0.9, 0.3], [0.0, 1.0]])
    assert geometry.structure_agreement(vectors, structural) == pytest.approx(1.0)
    assert np.isnan(geometry.structure_agreement(np.eye(3), structural))
