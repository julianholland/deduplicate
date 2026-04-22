import re

from deduplicate_lib.plugins.duplicate_detection_algorithms.multi_hashing import (
    MultiHashing,
)
from deduplicate_lib.plugins.duplicate_detection_algorithms.multi_hashing import fast_round_and_perturb
import numpy as np
import pytest
import copy

def test_fast_round_and_perturb_numba():
    input_vector = np.array([1.01, 2.01])
    perturbation_array = np.array([0.1, 0.2, 0.3])
    tolerance = 0.1

    result = fast_round_and_perturb(input_vector, perturbation_array, tolerance)
    expected_rounded_vector = np.round(input_vector / tolerance) * tolerance
    expected_result = np.zeros((perturbation_array.shape[0], len(input_vector)))
    for i in range(perturbation_array.shape[0]):
        expected_result[i] = expected_rounded_vector * perturbation_array[i]

    assert np.allclose(result, expected_result)


def test_str_representation():
    dm = MultiHashing(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0]]),
        distance_matrix=np.array([]),
    )
    assert str(dm) == "MultiHashing(tolerance=0.1, perturbations=200, sigma_acceptance_threshold=1)"

def test_set_perturbation_array(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.set_perturbation_array()
    assert dda.perturbation_array.shape[0] == dda.perturbations

def test_round_to_tolerance(multi_hashing_dda):
    dda = multi_hashing_dda
    rounded_vector = dda.round_to_tolerance()
    assert np.allclose(rounded_vector, np.array([1.0, 2.0]))

def test_create_hash_vector_array(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.set_perturbation_array()  
    hash_vector_array = dda.create_hash_vector_array()
    assert hash_vector_array.shape == (dda.max_vector_array_size, dda.perturbations)
    assert dda.get_filled_hash_vector_array().shape == (3, dda.perturbations)

def test_create_hash_vector_without_setting_perturbation_array(multi_hashing_dda):
    dda = multi_hashing_dda
    with pytest.warns(UserWarning, match="Perturbation array shape does not match expected number of perturbations; recomputing. \nAssign the perturbation array prior duplication checks to avoid this warning."):
        dda.create_hash_vector_array()
    assert dda.get_filled_hash_vector_array().shape == (3, dda.perturbations)

def test_create_hash_vector_array_consistency(multi_hashing_dda):
    dda1 = multi_hashing_dda
    dda1.set_perturbation_array()
    dda1.create_hash_vector_array()
    assert dda1.get_filled_hash_vector_array().shape == (3, dda1.perturbations)

    dda2 = copy.copy(dda1)
    hash_vector_array_1 = dda1.create_hash_vector_array()
    hash_vector_array_2 = dda2.create_hash_vector_array()
    assert np.array_equal(hash_vector_array_1, hash_vector_array_2)

def test_create_hash_vector_array_different_seeds(multi_hashing_dda):
    dda1 = copy.copy(multi_hashing_dda)
    dda1.seed = 803
    dda1.set_perturbation_array()  
    
    dda2 = copy.copy(multi_hashing_dda)
    dda2.seed = 8  # different seed
    dda2.set_perturbation_array()
    
    hash_vector_array_1 = dda1.create_hash_vector_array()
    hash_vector_array_2 = dda2.create_hash_vector_array()

    assert not np.array_equal(hash_vector_array_1, hash_vector_array_2)

def test_create_hash_vector_array_different_perturbations(multi_hashing_dda):
    dda1 = copy.copy(multi_hashing_dda)
    dda1.perturbations = 200
    dda1.set_perturbation_array()

    dda2 = copy.copy(multi_hashing_dda)
    dda2.perturbations = 300 # different number of perturbations
    dda2.set_perturbation_array()

    hash_vector_array_1 = dda1.create_hash_vector_array()
    hash_vector_array_2 = dda2.create_hash_vector_array()

    assert hash_vector_array_1.shape[1] == 200
    assert hash_vector_array_2.shape[1] == 300

def test_create_hash_vector_array_different_input_vectors(multi_hashing_dda):
    # should not effect the hash vector array as only the dataset vectors are used to create the hash vector array, not the input vector
    dda1 = copy.copy(multi_hashing_dda)
    dda1.set_perturbation_array()

    dda2 = copy.copy(multi_hashing_dda)
    dda2.input_vector = np.array([10.0, 20.0])
    dda2.set_perturbation_array()

    hash_vector_array_1 = dda1.create_hash_vector_array()
    hash_vector_array_2 = dda2.create_hash_vector_array()

    assert np.array_equal(hash_vector_array_1, hash_vector_array_2)

def test_create_hash_vector_array_different_dataset_arrays(multi_hashing_dda):
    dda1 = copy.copy(multi_hashing_dda)
    dda1.set_perturbation_array()

    dda2 = copy.copy(multi_hashing_dda)
    dda2.set_dataset_array(np.array([[10.0, 20.0], [11.0, 21.0], [9.0, 19.0]]))
    dda2.set_perturbation_array()

    hash_vector_array_1 = dda1.create_hash_vector_array()
    hash_vector_array_2 = dda2.create_hash_vector_array()

    assert not np.array_equal(hash_vector_array_1, hash_vector_array_2)

def test_duplicate_check(multi_hashing_dda):
    dda = multi_hashing_dda

    dda.set_perturbation_array()  # Ensure perturbation array is set before duplication check
    dda.create_hash_vector_array()  # Ensure hash vector array is created before duplication check
    assert dda.duplicate_check()

    dda.input_vector = np.array([16.0, 200.0])
    assert not dda.duplicate_check()


def test_duplicate_check_if_hash_vector_array_wrong_shape(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.hash_vector_array = np.zeros((1, 1), dtype=int)  # incorrect shape

    dda.set_perturbation_array()  # Ensure perturbation array is set before duplication check

    with pytest.warns(UserWarning, match="Hash vector array shape does not match dataset; recomputing."):
        duplicate_check = dda.duplicate_check()
    assert duplicate_check

def test_set_acceptance_threshold(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.sigma_accepatnce_threshold = 2
    dda.perturbations = 200
    dda.set_acceptance_threshold()
    assert dda.acceptance_threshold == 0.954499736103642
    dda.perturbations = 5

def test_set_acceptance_threshold_bad_sigma(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.sigma_accepatnce_threshold = 5  # not in the sigma_dict
    with pytest.raises(ValueError, match=r"Sigma acceptance threshold must be an integer between 1 and 4 inclusive."):
        dda.set_acceptance_threshold()

def test_set_acceptance_threshold_warning(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.sigma_accepatnce_threshold = 2
    dda.perturbations = 5
    with pytest.warns(UserWarning, match=re.escape("Sigma acceptance threshold of 2 corresponds to an acceptance threshold of 0.954499736103642 which may be too high for the number of perturbations (5) and could lead to false positives. Consider lowering the sigma acceptance threshold or increasing the number of perturbations to over 22.")):
        dda.set_acceptance_threshold()

def test_get_dataset_unique_structures(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.perturbations = 5
    dda.set_dataset_array(np.array([[1.0, 2.0], [1.001, 2.001], [10.0, 20.0]]))

    dda.set_perturbation_array()  
    dda.create_hash_vector_array()
    unique_count = dda.get_dataset_unique_structures()
    assert unique_count == 2

def test_distance_matrix_computation_with_hash_vector_array(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.perturbations = 5
    dda.set_perturbation_array()  
    dda.create_hash_vector_array()
    dda.compute_distance_matrix(dda.hash_vector_array)

def test_pre_dda_processing(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.pre_dda_processing()
    assert dda.perturbation_array.shape[0] == dda.perturbations

def test_add_input_vector_to_dda(multi_hashing_dda):
    dda = multi_hashing_dda
    initial_dataset_size = dda.get_filled_dataset_array().shape[0]
    dda.set_perturbation_array()  
    dda.create_hash_vector_array()
    dda.add_input_vector_to_dda()
    assert dda.get_filled_dataset_array().shape[0] == initial_dataset_size + 1
    assert np.array_equal(dda.get_filled_dataset_array()[-1], dda.input_vector)
    assert dda.get_filled_hash_vector_array().shape[0] == initial_dataset_size + 1

def test_get_uniqueness_score(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.set_perturbation_array()
    dda.create_hash_vector_array()
    dda.input_vector = dda.dataset_array[0]  # identical to first dataset vector, should be classified as duplicate
    uniqueness_score=dda.get_uniqueness_score()

    assert uniqueness_score == 0.0

    dda.input_vector = np.array([10.0, 20.0])  # different from all dataset vectors, should be classified as unique
    uniqueness_score=dda.get_uniqueness_score()
    assert uniqueness_score > 0.0 and uniqueness_score <= 1.0