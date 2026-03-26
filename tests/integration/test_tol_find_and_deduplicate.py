import pytest
import numpy as np
from deduplicate.plugins.tolerance_calculators.natural_tolerance_plateau_probe import (
    NaturalTolerancePlateauProbe,
)
from deduplicate.plugins.tolerance_calculators.perturbed_dataset_reclustering import (
    PerturbedDatasetReclustering,
)

rng=np.random.default_rng(803)
max_data_distance = 1.0
min_data_distance = 0.2
dimensions = 256
original_data_point = np.ones(dimensions) * max_data_distance/2
data = rng.uniform(
    low=min_data_distance, high=max_data_distance, size=(30, dimensions))


@pytest.mark.parametrize(
    "dda_fixture",
    [["distance_matrix_dda", 0.16824430025255346], ["multi_hashing_dda", 2.026742238406654]]
)
def test_pdr_with_dda(request, dda_fixture):
    dda = request.getfixturevalue(dda_fixture[0])
    dda.dataset_array = data

    tc = PerturbedDatasetReclustering(
        duplicate_detection_algorithm_object=dda,
        perturbations_per_vector=10,
        perturbation_scale=0.01,
        binary_search_steps=30,
    )
    tc.create_perturbed_dataset()
    tc.duplicate_detection_algorithm_object.pre_dda_processing(tc.tolerance_dataset_array)
    tolerance = tc.calculate_tolerance()
    print(tolerance)
    # assert False
    assert isinstance(tolerance, float)
    assert tolerance > 0.0
    assert np.isclose(tolerance, dda_fixture[1], atol=0.05 * dda_fixture[1])

@pytest.mark.parametrize(
    "dda_fixture",
    [["distance_matrix_dda", 0.17624324801178806], ["multi_hashing_dda", 0.784201633170273]]
)
def test_ntpp_with_dda(request, dda_fixture):
    dda = request.getfixturevalue(dda_fixture[0])
    dda.dataset_array = data

    tc = NaturalTolerancePlateauProbe(
        duplicate_detection_algorithm_object=dda,
        perturbations_per_vector=10,
        perturbation_scale=0.01,
        binary_search_steps=30,
        probe_steps=100,
    )
    
    tc.create_perturbed_dataset()
    tc.duplicate_detection_algorithm_object.pre_dda_processing(tc.tolerance_dataset_array)
    tolerance = tc.calculate_tolerance()
    print(tolerance)
    # assert False
    assert isinstance(tolerance, float)
    assert tolerance > 0.0
    assert np.isclose(tolerance, dda_fixture[1], atol=0.05 * dda_fixture[1])