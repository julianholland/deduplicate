import pytest
import numpy as np
from deduplicate_lib.plugins.tolerance_calculators.perturbed_dataset_reclustering import (
    PerturbedDatasetReclustering,
)


def test_str_representation(distance_matrix_dda):
    dda = distance_matrix_dda
    assert (
        str(
            PerturbedDatasetReclustering(
                duplicate_detection_algorithm_object=dda,
                perturbations_per_vector=1000,
                perturbation_scale=0.1,
            )
        )
        == f"PerturbedDatasetReclustering(perturbations_per_vector=1000, perturbation_scale=0.1, dda={str(dda).split('(')[0]})"
    )

@pytest.mark.parametrize(
    "dda_fixture",
    ["distance_matrix_dda", "multi_hashing_dda"]
)
def test_calculate_tolerance(request, dda_fixture):
    # create distance matrix dda
    dda = request.getfixturevalue(dda_fixture)

    # create perturbed dataset reclustering tolerance calculator with distance matrix dda
    tc = PerturbedDatasetReclustering(
        duplicate_detection_algorithm_object=dda,
        perturbations_per_vector=100,
        perturbation_scale=0.01,
        binary_search_steps=20,
    )

    # generate rattled data and compute distance matrix
    tc.create_perturbed_dataset()
    tc.duplicate_detection_algorithm_object.pre_dda_processing(tc.tolerance_dataset_array)

    # use these to calculate the tolerance
    tolerance = tc.calculate_tolerance()
    print(f"Calculated Average Tolerance: {tolerance}")
    # ensure the tolerance is a positive float and that it yields the correct number of unique vectors in the original dataset when applied to the perturbed dataset
    assert isinstance(tolerance, float)
    assert tolerance > 0.0
    tc.duplicate_detection_algorithm_object.tolerance = tolerance
    with tc.temp_attr(
        tc.duplicate_detection_algorithm_object,
        "dataset_array",
        tc.tolerance_dataset_array,
    ):
        unique_vectors = (
            tc.duplicate_detection_algorithm_object.get_dataset_unique_structures()
        )
    assert unique_vectors == len(tc.duplicate_detection_algorithm_object.dataset_array)

    # repeat for loose tolerance
    tc_loose = PerturbedDatasetReclustering(
        duplicate_detection_algorithm_object=dda,
        perturbations_per_vector=100,
        perturbation_scale=0.01,
        binary_search_steps=20,
        target_structures_threshold="loose",
    )
    tc_loose.create_perturbed_dataset()
    tc_loose.duplicate_detection_algorithm_object.pre_dda_processing(tc_loose.tolerance_dataset_array)
    

    # use these to calculate the tolerance
    loose_tolerance = tc_loose.calculate_tolerance()
    print(f"Calculated Loose Tolerance: {loose_tolerance}")

    assert isinstance(loose_tolerance, float)
    assert loose_tolerance > 0.0
    tc_loose.duplicate_detection_algorithm_object.tolerance = loose_tolerance
    with tc_loose.temp_attr(
        tc_loose.duplicate_detection_algorithm_object,
        "dataset_array",
        tc_loose.tolerance_dataset_array,
    ):
        unique_vectors = (
            tc_loose.duplicate_detection_algorithm_object.get_dataset_unique_structures()
        )
    assert unique_vectors == len(tc_loose.duplicate_detection_algorithm_object.dataset_array)

    assert loose_tolerance >= tolerance

    # repeat for tight tolerance
    tc_tight = PerturbedDatasetReclustering(
        duplicate_detection_algorithm_object=dda,
        perturbations_per_vector=100,
        perturbation_scale=0.01,
        binary_search_steps=20,
        target_structures_threshold="tight",
    )
    tc_tight.create_perturbed_dataset()
    tc_tight.duplicate_detection_algorithm_object.pre_dda_processing(tc_tight.tolerance_dataset_array)
    

    # use these to calculate the tolerance
    tight_tolerance = tc_tight.calculate_tolerance()
    print(f"Calculated Tight Tolerance: {tight_tolerance}")
    assert isinstance(tight_tolerance, float)
    assert tight_tolerance > 0.0
    tc_tight.duplicate_detection_algorithm_object.tolerance = tight_tolerance
    with tc_tight.temp_attr(
        tc_tight.duplicate_detection_algorithm_object,
        "dataset_array",
        tc_tight.tolerance_dataset_array,
    ):
        unique_vectors = (
            tc_tight.duplicate_detection_algorithm_object.get_dataset_unique_structures()
        )
    assert unique_vectors == len(tc_tight.duplicate_detection_algorithm_object.dataset_array)

    assert tight_tolerance <= tolerance
    assert tight_tolerance <= loose_tolerance

    tc.binary_search_steps = 1
    original_best_max = (
        np.ptp(tc.tolerance_dataset_array)
        + np.mean(np.std(tc.tolerance_dataset_array, axis=0))
    ) * tc.duplicate_detection_algorithm_object.dataset_array.shape[1]
    with pytest.warns(
        UserWarning,
        match=f"No exact tolerance found for target of {len(dda.dataset_array)} unique vectors during binary search.\n Returning closest tolerance found: {original_best_max} with 1 unique vectors.",
    ):
        tolerance = tc.calculate_tolerance()

    # assert False
