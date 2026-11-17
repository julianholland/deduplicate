import dataclasses
import re

import numpy as np
import pytest

from deduplicate_lib.plugins.duplicate_detection_algorithms.distance_matrix import (
    DistanceMatrix,
)
from deduplicate_lib.plugins.tolerance_calculators.natural_tolerance_plateau_probe import (
    NaturalTolerancePlateauProbe,
    Plateau,
)


def geometric_results(counts):
    """``{tolerance: count}`` with log-spaced tolerances from 0.01 to 1."""
    return {float(t): c for t, c in zip(np.geomspace(0.01, 1, len(counts)), counts)}


@pytest.fixture
def tc(distance_matrix_dda):
    return NaturalTolerancePlateauProbe(
        distance_matrix_dda,
        datapoints_to_calculate_gradient=2,
        plateau_threshold=0.1,
    )


def multi_scale_data():
    """200 rows: 10 super-clusters x 5 clusters (sigma 0.05) x 4 near-duplicates (sigma 0.001)."""
    rng = np.random.default_rng(1)
    centres = rng.uniform(0, 10, size=(10, 8))
    rows = []
    for c in centres:
        for cc in c + rng.normal(0, 0.05, size=(5, 8)):
            rows.extend(cc + rng.normal(0, 0.001, size=(4, 8)))
    return np.array(rows)


# --- construction ---


def test_str_representation(distance_matrix_dda):
    tc = NaturalTolerancePlateauProbe(
        distance_matrix_dda, probe_steps=40, sample_size=300, sampling="head"
    )
    assert (
        str(tc)
        == f"NaturalTolerancePlateauProbe(probe_steps=40, sample_size=300, sampling='head', dda={str(distance_matrix_dda).split('(')[0]})"
    )


def test_defaults(distance_matrix_dda):
    tc = NaturalTolerancePlateauProbe(distance_matrix_dda)
    assert tc.probe_steps == 100
    assert tc.datapoints_to_calculate_gradient is None
    assert tc.gradient_window() == 10  # 10% of the 100 probe steps
    assert tc.plateau_threshold == 1e-3
    assert tc.sample_size == 2000
    assert tc.sampling == "blocks"
    assert tc.binary_search_steps == 50


@pytest.mark.parametrize(
    "probe_steps,n_points,expected",
    [(100, None, 10), (200, None, 20), (400, None, 40), (10, None, 2), (100, 50, 5)],
)
def test_default_gradient_window_is_a_share_of_the_sweep(
    distance_matrix_dda, probe_steps, n_points, expected
):
    tc = NaturalTolerancePlateauProbe(distance_matrix_dda, probe_steps=probe_steps)
    assert tc.gradient_window(n_points) == expected


def test_explicit_gradient_window_overrides_default(distance_matrix_dda):
    tc = NaturalTolerancePlateauProbe(
        distance_matrix_dda, probe_steps=400, datapoints_to_calculate_gradient=3
    )
    assert tc.gradient_window() == 3
    assert tc.gradient_window(50) == 3


def test_options_are_keyword_only(distance_matrix_dda):
    with pytest.raises(TypeError):
        NaturalTolerancePlateauProbe(distance_matrix_dda, 100)  # type: ignore


def test_plateau_is_frozen():
    plateau = Plateau(0.1, 0.2, 0.15, 3, 4, False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plateau.trivial = True  # type: ignore


# --- tolerance_probe / get_plateau_log ---


def test_tolerance_probe(tc, distance_matrix_dda):
    data = np.random.default_rng(0).uniform(0, 1, size=(20, 4))
    tuning = distance_matrix_dda.tuning_copy(data)
    results = tc.tolerance_probe(tuning, 0.01, 10.0, 10)
    assert len(results) == 10
    tols = list(results)
    assert np.allclose(np.log(tols[1:]) - np.log(tols[:-1]), np.log(10.0 / 0.01) / 9)
    assert all(isinstance(u, int) for u in results.values())
    counts = list(results.values())
    assert counts == sorted(counts, reverse=True)  # larger tolerance, fewer unique
    assert counts[0] == 20
    assert counts[-1] == 1


def test_tolerance_probe_swaps_data_via_fresh_copy(tc, distance_matrix_dda):
    """Regression: a stale distance matrix must never be reused for new data."""
    distance_matrix_dda.pre_dda_processing()
    far_apart = np.array([[0.0, 0.0], [100.0, 100.0], [200.0, 200.0]])
    tuning = distance_matrix_dda.tuning_copy(far_apart)
    results = tc.tolerance_probe(tuning, 0.1, 0.9, 3)
    assert all(count == 3 for count in results.values())


def test_tolerance_probe_repairs_bad_lower_bound(tc, distance_matrix_dda):
    tuning = distance_matrix_dda.tuning_copy(np.array([[0.0, 0.0], [1.0, 1.0]]))
    results = tc.tolerance_probe(tuning, 0.0, 1.0, 5)
    assert min(results) == pytest.approx(1e-3)
    assert max(results) == pytest.approx(1.0)


def test_get_plateau_log(tc):
    results = geometric_results([1] * 10 + [2] * 10 + [3] * 10)
    sorted_tols = sorted(results)
    log = tc.get_plateau_log(sorted_tols, results)
    assert isinstance(log, np.ndarray)
    assert log.dtype == bool
    assert len(log) == len(sorted_tols) - tc.datapoints_to_calculate_gradient
    assert set(tc.plateau_data) == {"tolerances", "unique_counts", "gradients"}


def test_get_plateau_log_validates_window(tc):
    results = geometric_results(list(range(10)))
    tc.datapoints_to_calculate_gradient = 1
    with pytest.raises(
        ValueError,
        match=re.escape("datapoints_to_calculate_gradient must be greater than 1."),
    ):
        tc.get_plateau_log(sorted(results), results)
    tc.datapoints_to_calculate_gradient = 10
    with pytest.raises(ValueError, match="Not enough tolerance steps"):
        tc.get_plateau_log(sorted(results), results)


# --- find_plateaus on hand-made sweeps ---


def test_find_plateaus_no_plateaus(tc):
    assert tc.find_plateaus(geometric_results(list(range(20, 0, -1)))) == []


def test_find_plateaus_at_the_very_start(tc):
    plateaus = tc.find_plateaus(geometric_results([20] * 6 + [15, 10, 5, 1]))
    assert len(plateaus) == 1
    (plateau,) = plateaus
    assert plateau.start_tolerance == pytest.approx(0.01)  # first probed tolerance
    assert plateau.unique_vectors == 20
    assert plateau.trivial  # every vector unique
    assert plateau.start_tolerance < plateau.tolerance < plateau.end_tolerance


def test_find_plateaus_at_the_very_end(tc):
    plateaus = tc.find_plateaus(geometric_results([20, 15, 10, 8, 5] + [3] * 6))
    assert len(plateaus) == 1
    (plateau,) = plateaus
    assert plateau.end_tolerance == pytest.approx(1.0)  # last probed tolerance
    assert plateau.unique_vectors == 3
    assert not plateau.trivial  # not collapsed to a single vector


def test_find_plateaus_trivial_all_same_end(tc):
    plateaus = tc.find_plateaus(geometric_results([20, 15, 10, 8, 5] + [1] * 6))
    assert [p.trivial for p in plateaus] == [True]
    assert plateaus[0].unique_vectors == 1


def test_find_plateaus_true_false_true_pattern(tc):
    plateaus = tc.find_plateaus(geometric_results([20] * 5 + [10, 5] + [3] * 5))
    assert len(plateaus) == 2
    first, last = plateaus
    assert first.start_tolerance == pytest.approx(0.01)
    assert last.end_tolerance == pytest.approx(1.0)
    assert first.end_tolerance < last.start_tolerance
    assert (first.unique_vectors, last.unique_vectors) == (20, 3)
    assert (first.trivial, last.trivial) == (True, False)


def test_find_plateaus_merges_equal_count_runs(tc):
    # a one-point bump (11) inside the 10-plateau must not split it in two
    counts = [20, 17, 14] + [10] * 3 + [11] + [10] * 3 + [5, 2, 1]
    plateaus = tc.find_plateaus(geometric_results(counts))
    assert [p.unique_vectors for p in plateaus] == [10]
    assert plateaus[0].steps >= 7


def test_find_plateaus_keeps_runs_with_different_counts_apart(tc):
    counts = [20] + [10] * 5 + [5] * 5 + [1]
    plateaus = tc.find_plateaus(geometric_results(counts))
    assert [p.unique_vectors for p in plateaus] == [10, 5]
    assert plateaus[0].end_tolerance < plateaus[1].start_tolerance or (
        plateaus[0].end_tolerance <= plateaus[1].start_tolerance
    )


def test_find_plateaus_uses_max_unique_vectors_for_triviality(tc):
    results = geometric_results([20] * 6 + [15, 10, 5, 1])
    tc.max_unique_vectors = 50  # 20 vectors is not "all unique" any more
    assert [p.trivial for p in tc.find_plateaus(results)] == [False]


# --- full runs ---


def test_probe_validates_settings(tc):
    tc.probe_steps = 2
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Not enough tolerance steps to calculate gradient with the given datapoints_to_calculate_gradient."
        ),
    ):
        tc.calculate_tolerance()
    tc.datapoints_to_calculate_gradient = 1
    with pytest.raises(ValueError, match="must be greater than 1"):
        tc.probe()


@pytest.mark.parametrize("dda_fixture", ["distance_matrix_dda", "multi_hashing_dda"])
def test_calculate_tolerance_runs_and_leaves_dda_alone(request, dda_fixture):
    dda = request.getfixturevalue(dda_fixture)
    data = multi_scale_data()
    dda.set_dataset_array(data)
    tolerance_before = dda.tolerance
    tc = NaturalTolerancePlateauProbe(dda, probe_steps=60, binary_search_steps=30)
    tolerance = tc.calculate_tolerance()
    assert isinstance(tolerance, float)
    assert tolerance > 0.0
    assert tc.all_different_tolerance < tc.all_same_tolerance
    assert dda.tolerance == tolerance_before
    assert np.array_equal(dda.get_filled_dataset_array(), data)
    assert len(tc.probe_results) == 60


def test_calculate_tolerance_falls_back_without_plateaus(distance_matrix_dda):
    data = np.random.default_rng(3).uniform(0, 1, size=(30, 6))
    distance_matrix_dda.set_dataset_array(data)
    tc = NaturalTolerancePlateauProbe(
        distance_matrix_dda,
        probe_steps=5,
        datapoints_to_calculate_gradient=3,
        plateau_threshold=1e-6,
    )
    with pytest.warns(UserWarning, match="No natural plateaus found"):
        tolerance = tc.calculate_tolerance()
    assert tolerance == pytest.approx(
        np.sqrt(tc.all_different_tolerance * tc.all_same_tolerance)
    )


def test_find_natural_plateaus_runs_the_probe_if_needed(distance_matrix_dda):
    distance_matrix_dda.set_dataset_array(multi_scale_data())
    tc = NaturalTolerancePlateauProbe(distance_matrix_dda, probe_steps=60)
    assert tc.probe_results == {}
    plateaus = tc.find_natural_plateaus()
    assert tc.probe_results
    assert all(isinstance(p, Plateau) and not p.trivial for p in plateaus)


# --- multi-scale data with two known clustering scales ---


@pytest.fixture(scope="module")
def multi_scale_probe():
    dda = DistanceMatrix(dataset_array=multi_scale_data(), max_vector_array_size=200)
    tc = NaturalTolerancePlateauProbe(
        dda, probe_steps=200, datapoints_to_calculate_gradient=6
    )
    tc.probe()
    return tc


def plateau_with_count(plateaus, count):
    (match,) = [p for p in plateaus if p.unique_vectors == count]
    return match


def test_multi_scale_plateaus(multi_scale_probe):
    plateaus = multi_scale_probe.find_natural_plateaus(include_trivial=True)
    assert plateaus[0].trivial
    assert plateaus[0].unique_vectors == 200
    assert plateaus[-1].trivial
    assert plateaus[-1].unique_vectors == 1
    non_trivial = [p for p in plateaus if not p.trivial]
    assert {50, 10} <= {p.unique_vectors for p in non_trivial}
    assert [p.unique_vectors for p in non_trivial] == sorted(
        (p.unique_vectors for p in non_trivial), reverse=True
    )
    assert multi_scale_probe.find_natural_plateaus() == non_trivial


@pytest.mark.parametrize(
    ("condition", "count"), [("tightest", 50), ("longest", 10), ("loosest", 10)]
)
def test_multi_scale_conditions(multi_scale_probe, condition, count):
    tc = multi_scale_probe
    plateau = plateau_with_count(tc.find_natural_plateaus(), count)
    tolerance = tc.calculate_tolerance(condition=condition)
    assert plateau.start_tolerance <= tolerance <= plateau.end_tolerance
    # and that tolerance really finds that many unique vectors
    dda = tc.duplicate_detection_algorithm_object.tuning_copy(tc.tuning_array)
    dda.tolerance = tolerance
    assert dda.get_dataset_unique_structures() == count


def test_invalid_condition(multi_scale_probe):
    with pytest.raises(ValueError, match="Invalid condition 'sideways'"):
        multi_scale_probe.calculate_tolerance(condition="sideways")
