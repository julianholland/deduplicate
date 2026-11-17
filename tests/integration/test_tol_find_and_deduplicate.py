import numpy as np
import pytest

from deduplicate_lib.plugins.duplicate_detection_algorithms.distance_matrix import (
    DistanceMatrix,
)
from deduplicate_lib.plugins.duplicate_detection_algorithms.multi_hashing import (
    MultiHashing,
)
from deduplicate_lib.plugins.tolerance_calculators.natural_tolerance_plateau_probe import (
    NaturalTolerancePlateauProbe,
)
from deduplicate_lib.plugins.tolerance_calculators.perturbed_dataset_reclustering import (
    PerturbedDatasetReclustering,
)

rng = np.random.default_rng(803)
max_data_distance = 1.0
dimensions = 256
original_data_point = np.ones(dimensions) * max_data_distance / 2
big_random_data = rng.uniform(low=0.0, high=max_data_distance, size=(30, dimensions))

small_random_data = rng.uniform(
    low=0.0, high=max_data_distance / 100, size=(30, dimensions)
)

random_point = rng.integers(low=0, high=30)
perturbed_data = np.array(
    [big_random_data[random_point] + small_random_data[x] for x in range(30)]
)

data = np.vstack([original_data_point, perturbed_data])


def assert_keeps_one_vector_per_group(tolerance):
    """data is one point plus a 30-vector cluster: the found tolerance must
    keep exactly the point and the cluster's first member."""
    dda = DistanceMatrix(dataset_array=data, max_vector_array_size=len(data))
    dda.tolerance = tolerance
    dda.get_dataset_unique_structures()
    assert list(dda.get_unique_vector_indices()) == [0, 1]


def assert_multi_hashing_dedup_is_sensible(tolerance):
    """MultiHashing is approximate: it must drop most of the 30-vector cluster
    but keep the lone point and at least one cluster member."""
    dda = MultiHashing(
        dataset_array=data,
        max_vector_array_size=len(data),
        tolerance=tolerance,
        perturbations=200,
    )
    n_unique = dda.get_dataset_unique_structures()
    assert 2 <= n_unique < len(data) // 2
    assert dda.get_unique_vector_mask()[0]  # the lone point is always kept


@pytest.mark.parametrize("dda_fixture", ["distance_matrix_dda", "multi_hashing_dda"])
def test_pdr_with_dda(request, dda_fixture):
    dda = request.getfixturevalue(dda_fixture)
    dda.set_dataset_array(data)
    dda.input_vector = np.array([])
    tc = PerturbedDatasetReclustering(
        dda,
        perturbations_per_vector=10,
        perturbation_scale=0.01,
        binary_search_steps=30,
        sample_size=len(data),
        sampling="head",
    )

    tolerance = tc.calculate_tolerance()

    assert isinstance(tolerance, float)
    assert tolerance > 0.0
    if dda_fixture == "distance_matrix_dda":
        assert_keeps_one_vector_per_group(tolerance)
    else:
        assert_multi_hashing_dedup_is_sensible(tolerance)


@pytest.mark.parametrize("dda_fixture", ["distance_matrix_dda", "multi_hashing_dda"])
def test_ntpp_with_dda(request, dda_fixture):
    dda = request.getfixturevalue(dda_fixture)
    dda.set_dataset_array(data)
    dda.input_vector = np.array([])

    tc = NaturalTolerancePlateauProbe(
        dda,
        binary_search_steps=30,
        probe_steps=200,
        sample_size=len(data),
        sampling="head",
    )
    tolerance = tc.calculate_tolerance()

    assert isinstance(tolerance, float)
    assert tolerance > 0.0
    if dda_fixture == "distance_matrix_dda":
        assert_keeps_one_vector_per_group(tolerance)
    else:
        assert_multi_hashing_dedup_is_sensible(tolerance)
