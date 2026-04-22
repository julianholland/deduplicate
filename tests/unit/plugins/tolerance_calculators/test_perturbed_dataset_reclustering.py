import pytest
import numpy as np
from deduplicate_lib.plugins.tolerance_calculators.perturbed_dataset_reclustering import (
    PerturbedDatasetReclustering,
)
import re

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
    def pdr_with_threshold_type(threshold_type):
        tc = PerturbedDatasetReclustering(
            duplicate_detection_algorithm_object=dda,
            perturbations_per_vector=100,
            perturbation_scale=0.01,
            binary_search_steps=20,
            target_unique_vectors_threshold=threshold_type,
        )

        # generate rattled data and compute distance matrix
        tc.create_perturbed_dataset()
    
        # use these to calculate the tolerance
        tolerance = tc.calculate_tolerance()
        print(f"Calculated {threshold_type} tolerance: {tolerance}")
        # ensure the tolerance is a positive float and that it yields the correct number of unique vectors in the original dataset when applied to the perturbed dataset
        assert isinstance(tolerance, float)
        assert tolerance > 0.0
        tc.duplicate_detection_algorithm_object.tolerance = tolerance
        old_dataset_array = tc.duplicate_detection_algorithm_object.get_filled_dataset_array()
        tc.duplicate_detection_algorithm_object.set_dataset_array(tc.tolerance_dataset_array)
        unique_vectors = (
            tc.duplicate_detection_algorithm_object.get_dataset_unique_structures()
        )
        tc.duplicate_detection_algorithm_object.set_dataset_array(old_dataset_array)
        assert unique_vectors == len(tc.duplicate_detection_algorithm_object.get_filled_dataset_array())

        return tolerance

    average_tolerance = pdr_with_threshold_type("average")
    loose_tolerance = pdr_with_threshold_type("loose")
    tight_tolerance = pdr_with_threshold_type("tight")

    assert loose_tolerance > average_tolerance
    assert tight_tolerance < average_tolerance
    assert tight_tolerance < loose_tolerance

def test_initialize_with_target_unique_vectors(distance_matrix_dda):
    dda = distance_matrix_dda
    target_unique_vectors = 2
    tc = PerturbedDatasetReclustering(
        duplicate_detection_algorithm_object=dda,
        perturbations_per_vector=100,
        perturbation_scale=0.01,
        binary_search_steps=20,
        target_unique_vectors=target_unique_vectors,
    )
    assert tc.target_unique_vectors == target_unique_vectors

@pytest.mark.parametrize(
    "dda_fixture",
    ["distance_matrix_dda", "multi_hashing_dda"]
)
def test_calculate_tolerance_with_insufficient_binary_search_steps(request, dda_fixture):
    # create distance matrix dda
    dda = request.getfixturevalue(dda_fixture)

    # create perturbed dataset reclustering tolerance calculator with distance matrix dda
    tc = PerturbedDatasetReclustering(
        duplicate_detection_algorithm_object=dda,
        perturbations_per_vector=100,
        perturbation_scale=0.01,
        binary_search_steps=1,
    )
    tc.create_perturbed_dataset()

    original_best_max = (
        np.ptp(tc.tolerance_dataset_array)
        + np.mean(np.std(tc.tolerance_dataset_array, axis=0))
    ) * tc.duplicate_detection_algorithm_object.dataset_array.shape[1]
    
    with pytest.warns(
        UserWarning,
        match=(
            rf"No exact tolerance found for target of {len(dda.get_filled_dataset_array())} unique vectors during binary search\.\n"
            rf" Returning closest tolerance found: {re.escape(str(original_best_max))} with \d+ unique vectors\."
        ),
    ):
        print(tc.calculate_tolerance())
   

def test_calculate_tolerance_raises_for_invalid_target_unique_vectors_threshold(distance_matrix_dda, monkeypatch):
    tc = PerturbedDatasetReclustering(
        duplicate_detection_algorithm_object=distance_matrix_dda,
        target_unique_vectors_threshold="invalid_mode",
    )

    # Avoid unrelated setup behavior; we only want to test the threshold validation branch
    monkeypatch.setattr(tc, "_ensure_perturbed_dataset", lambda: None)

    with pytest.raises(
        ValueError,
        match=r"Invalid target_unique_vectors_threshold: invalid_mode\. Must be 'average', 'loose', or 'tight'\.",
    ):
        tc.calculate_tolerance()