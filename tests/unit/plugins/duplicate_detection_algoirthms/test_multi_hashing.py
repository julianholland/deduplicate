from deduplicate.plugins.duplicate_detection_algorithms.multi_hashing import (
    MultiHashing,
)
import numpy as np
import pytest

def test_str_representation():
    dm = MultiHashing(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0]]),
        distance_matrix=np.array([]),
    )
    assert str(dm) == "MultiHashing(tolerance=0.1, perturbations=200, acceptance_threshold=0.5)"

def test_set_perturbation_array():
    dda = MultiHashing(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0]]),
        distance_matrix=np.array([]),
        seed=803,
    )
    dda.set_perturbation_array()
    assert dda.perturbation_array.shape[0] == dda.perturbations

def test_round_to_tolerance():
    dda = MultiHashing(
        tolerance=0.1,
        input_vector=np.array([1.05, 2.03]),
        dataset_array=np.array([[1.0, 2.0]]),
        distance_matrix=np.array([]),
    )
    rounded_vector = dda.round_to_tolerance()
    assert np.allclose(rounded_vector, np.array([1.0, 2.0]))

def test_create_hash_vector_array():
    dda = MultiHashing(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
        distance_matrix=np.array([]),
    )
    dda.set_perturbation_array()  
    hash_vector_array = dda.create_hash_vector_array()
    assert hash_vector_array.shape == (3, dda.perturbations)

def test_create_hash_vector_without_setting_perturbation_array():
    dda = MultiHashing(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
        distance_matrix=np.array([]),
    )
    with pytest.warns(UserWarning, match="Perturbation array shape does not match expected number of perturbations; recomputing. \nAssign the perturbation array prior duplication checks to avoid this warning."):
        hash_vector_array = dda.create_hash_vector_array()
    assert hash_vector_array.shape == (3, dda.perturbations)

def test_create_hash_vector_array_consistency():
    dda = MultiHashing(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
        distance_matrix=np.array([]),
        seed=803,
    )
    dda.set_perturbation_array()
    hash_vector_array_1 = dda.create_hash_vector_array()
    hash_vector_array_2 = dda.create_hash_vector_array()
    assert np.array_equal(hash_vector_array_1, hash_vector_array_2)

def test_create_hash_vector_array_different_seeds():
    dda1 = MultiHashing(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
        distance_matrix=np.array([]),
        seed=803,
    )
    dda1.set_perturbation_array()  
    dda2 = MultiHashing(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
        distance_matrix=np.array([]),
        seed=804,
    )
    dda2.set_perturbation_array()
    hash_vector_array_1 = dda1.create_hash_vector_array()
    hash_vector_array_2 = dda2.create_hash_vector_array()
    assert not np.array_equal(hash_vector_array_1, hash_vector_array_2)

def test_create_hash_vector_array_different_perturbations():
    dda1 = MultiHashing(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
        distance_matrix=np.array([]),
        perturbations=200,
    )
    dda1.set_perturbation_array()
    dda2 = MultiHashing(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
        distance_matrix=np.array([]),
        perturbations=300,
    )
    dda2.set_perturbation_array()
    hash_vector_array_1 = dda1.create_hash_vector_array()
    hash_vector_array_2 = dda2.create_hash_vector_array()
    assert hash_vector_array_1.shape[1] == 200
    assert hash_vector_array_2.shape[1] == 300

def test_create_hash_vector_array_different_input_vectors():
    # should not effect the hash vector array as only the dataset vectors are used to create the hash vector array, not the input vector
    dda1 = MultiHashing(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
        distance_matrix=np.array([]),
        seed=803,
    )
    dda1.set_perturbation_array()
    dda2 = MultiHashing(
        tolerance=0.1,
        input_vector=np.array([10.0, 20.0]),
        dataset_array=np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
        distance_matrix=np.array([]),
        seed=803,
    )
    dda2.set_perturbation_array()
    hash_vector_array_1 = dda1.create_hash_vector_array()
    hash_vector_array_2 = dda2.create_hash_vector_array()
    assert np.array_equal(hash_vector_array_1, hash_vector_array_2)

def test_create_hash_vector_array_different_dataset_arrays():
    dda1 = MultiHashing(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
        distance_matrix=np.array([]),
        seed=803,
    )
    dda2 = MultiHashing(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[10.0, 20.0], [11.0, 21.0], [9.0, 19.0]]),
        distance_matrix=np.array([]),
        seed=803,
    )
    dda1.set_perturbation_array()
    dda2.set_perturbation_array()
    hash_vector_array_1 = dda1.create_hash_vector_array()
    hash_vector_array_2 = dda2.create_hash_vector_array()
    assert not np.array_equal(hash_vector_array_1, hash_vector_array_2)


def test_duplicate_check():
    dda = MultiHashing(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
    )
    dda.set_perturbation_array()  # Ensure perturbation array is set before duplication check
    dda.create_hash_vector_array()  # Ensure hash vector array is created before duplication check
    assert dda.duplicate_check()

    dda.input_vector = np.array([10.0, 20.0])
    assert not dda.duplicate_check()


def test_duplicate_check_if_hash_vector_array_wrong_shape():
    dda = MultiHashing(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
        hash_vector_array=np.array([1.0]),  # will be recomputed
    )
    dda.set_perturbation_array()  # Ensure perturbation array is set before duplication check
    with pytest.warns(UserWarning, match="Hash vector array shape does not match dataset; recomputing."):
        duplicate_check = dda.duplicate_check()
    assert duplicate_check
    

def test_get_dataset_unique_structures():
    
    dda = MultiHashing(
        tolerance=0.1,
        input_vector=np.array([0.0, 0.0]), # not used here
        dataset_array=np.array([[1.0, 2.0], [1.01, 2.01], [10.0, 20.0]]), # two similar one different
        perturbations=5, # sufficient to distinguish the two similar ones from the different one
    )
    dda.set_perturbation_array()  
    dda.create_hash_vector_array()
    unique_count = dda.get_dataset_unique_structures()
    assert unique_count == 2

def test_distance_matrix_computation_with_hash_vector_array():
    dda = MultiHashing(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
        perturbations=5,
    )
    dda.set_perturbation_array()  
    dda.create_hash_vector_array()
    dda.compute_distance_matrix(dda.hash_vector_array)