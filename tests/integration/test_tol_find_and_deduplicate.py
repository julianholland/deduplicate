import pytest
import numpy as np
from deduplicate_lib.plugins.duplicate_detection_algorithms.distance_matrix import (
    DistanceMatrix,
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


@pytest.mark.parametrize(
    "dda_fixture",
    [
        ["distance_matrix_dda", 0.18096144110600776],
        ["multi_hashing_dda", 0.2540022716601484],
    ],
)
def test_pdr_with_dda(request, dda_fixture):
    dda = request.getfixturevalue(dda_fixture[0])
    dda.set_dataset_array(data)
    dda.input_vector = np.array([])
    print(dda.dataset_array.shape)
    print(dda.vector_count)
    tc = PerturbedDatasetReclustering(
        duplicate_detection_algorithm_object=dda,
        perturbations_per_vector=10,
        perturbation_scale=0.01,
        binary_search_steps=30,
    )
    print(
        "pre perturbed dataset creation:",
        tc.tolerance_dataset_array.shape,
        tc.duplicate_detection_algorithm_object.dataset_array.shape,
        tc.duplicate_detection_algorithm_object.vector_count,
    )
    tc.create_perturbed_dataset()
    print(
        "post perturbed dataset creation:",
        tc.tolerance_dataset_array.shape,
        tc.duplicate_detection_algorithm_object.dataset_array.shape,
        tc.duplicate_detection_algorithm_object.vector_count,
    )

    tolerance = tc.calculate_tolerance()
    print(tolerance)

    assert isinstance(tolerance, float)
    assert tolerance > 0.0
    assert np.isclose(tolerance, dda_fixture[1], atol=0.05 * dda_fixture[1])
    if dda_fixture[0] == "distance_matrix_dda":
        assert_keeps_one_vector_per_group(tolerance)


@pytest.mark.parametrize(
    "dda_fixture,tolerance_result_fixture",
    [
        ("distance_matrix_dda", 0.1528962435655764),
        ("multi_hashing_dda", 0.10502871363961308),
    ],
)
def test_ntpp_with_dda(request, dda_fixture, tolerance_result_fixture):
    dda = request.getfixturevalue(dda_fixture)
    dda.set_dataset_array(data)
    dda.input_vector = np.array([])

    tc = NaturalTolerancePlateauProbe(
        duplicate_detection_algorithm_object=dda,
        perturbations_per_vector=10,
        perturbation_scale=0.01,
        binary_search_steps=30,
        probe_steps=200,
    )

    tc.create_perturbed_dataset()
    tolerance = tc.calculate_tolerance()
    print(tolerance)
    # assert False
    assert isinstance(tolerance, float)
    assert tolerance > 0.0

    assert np.isclose(
        tolerance, tolerance_result_fixture, atol=0.05 * tolerance_result_fixture
    )
    if dda_fixture == "distance_matrix_dda":
        assert_keeps_one_vector_per_group(tolerance)
