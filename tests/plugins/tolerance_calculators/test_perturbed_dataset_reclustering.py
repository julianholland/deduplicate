import pytest
import numpy as np
from deduplicate.plugins.tolerance_calculators.perturbed_dataset_reclustering import (
    PerturbedDatasetReclustering,
)
from deduplicate.plugins.duplicate_detection_algorithms.distance_matrix import (
    DistanceMatrix,
)
from deduplicate.plugins.duplicate_detection_algorithms.multi_hashing import (
    MultiHashing,
)


def test_str_representation():
    dda = DistanceMatrix(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[0.1, 0.2], [1.01, 2.01], [0.0, 20.0]]),
    )
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


def test_calculate_tolerance_distance_matrix():
    # create distance matrix dda
    dda = DistanceMatrix(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[0.1, 0.2], [0.5, 0.6], [1.8, 0.9]]),
    )

    # create perturbed dataset reclustering tolerance calculator with distance matrix dda
    tc = PerturbedDatasetReclustering(
        duplicate_detection_algorithm_object=dda,
        perturbations_per_vector=100,
        perturbation_scale=0.01,
        binary_search_steps=20,
    )

    # generate rattled data and compute distance matrix
    tc.create_perturbed_dataset()
    with tc.temp_attr(
        tc.duplicate_detection_algorithm_object,
        "dataset_array",
        tc.tolerance_dataset_array,
    ):
        tc.duplicate_detection_algorithm_object.pre_dda_processing()

    # use these to calculate the tolerance
    tolerance = tc.calculate_tolerance()

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


def test_calculate_tolerance_multi_hashing():
    # set up multi hashing dda
    dda = MultiHashing(
        tolerance=0.01,
        perturbations=5,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[0.1, 0.2], [0.5, 0.6], [1.8, 0.9]]),
    )

    # set up perturbed dataset reclustering tolerance calculator with multi hashing dda
    tc = PerturbedDatasetReclustering(
        duplicate_detection_algorithm_object=dda,
        perturbations_per_vector=10,
        perturbation_scale=0.01,
        binary_search_steps=20,
    )

    # create perturbed data and perturbation array for multi hashing
    tc.create_perturbed_dataset()
    tc.duplicate_detection_algorithm_object.pre_dda_processing()
    
    tolerance = tc.calculate_tolerance()
    
    tc.duplicate_detection_algorithm_object.tolerance = tolerance
    unique_vectors = (
        tc.duplicate_detection_algorithm_object.get_dataset_unique_structures()
    )
    
    assert isinstance(tolerance, float)
    assert tolerance > 0.0


    print("tolerance:", tolerance)
    
    assert unique_vectors == len(tc.duplicate_detection_algorithm_object.dataset_array)
    

    tc.binary_search_steps = 1
    
    original_best_max = (
        np.ptp(tc.tolerance_dataset_array)
        + np.mean(np.std(tc.tolerance_dataset_array, axis=0))
    ) * tc.duplicate_detection_algorithm_object.dataset_array.shape[1]
    # set the tolerance dataset array to be the same as the original dataset to ensure that the closest tolerance found is the one that yields the correct number of unique vectors
    with pytest.warns(
        UserWarning,
        match=f"No exact tolerance found for target of {len(dda.dataset_array)} unique vectors during binary search.\n Returning closest tolerance found: {original_best_max} with 1 unique vectors.",
    ):
        tc.calculate_tolerance()
