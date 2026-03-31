from deduplicate_lib.plugins.duplicate_detection_algorithms.distance_matrix import (
    DistanceMatrix,
)
import numpy as np
import pytest

def test_str_representation():
    dda = DistanceMatrix(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0]]),
        distance_matrix=np.array([]),
        distance_metric="euclidean",
    )
    assert str(dda) == "DistanceMatrix(tolerance=0.1, distance_metric=euclidean)"

def test_duplicate_check():
    dda = DistanceMatrix(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
        distance_matrix=np.array([]),  # not used here
        distance_metric="euclidean",
    )
    assert dda.duplicate_check()

    dda.input_vector = np.array([10.0, 20.0])
    assert not dda.duplicate_check()

def test_duplicate_check_if_distance_matrix_wrong_shape():
    dda = DistanceMatrix(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
        distance_matrix=np.array([1.0]),  # will be computed
        distance_metric="euclidean",
    )
    pytest.warns(UserWarning, match="Distance matrix shape does not match dataset; recomputing.")
    assert dda.duplicate_check()

def test_get_dataset_unique_structures():
    
    dm=np.array([[0.00000000e+00, 1.41421356e-02, 2.01246118e+01],
                 [1.41421356e-02, 0.00000000e+00, 2.01111959e+01],
                 [2.01246118e+01, 2.01111959e+01, 0.00000000e+00]]) # precomputed distance matrix for the dataset_array below
    
    dda = DistanceMatrix(
        tolerance=0.1,
        input_vector=np.array([0.0, 0.0]), # not used here
        dataset_array=np.array([[1.0, 2.0], [1.01, 2.01], [10.0, 20.0]]),
        distance_matrix=dm,
        distance_metric="euclidean",
    )
    unique_count = dda.get_dataset_unique_structures()
    print(dda.distance_matrix > 0.1)
    assert unique_count == 2

def test_ensure_distance_matrix(distance_matrix_dda):
    dda = distance_matrix_dda
    dda.compute_distance_matrix(dda.dataset_array)
    assert dda._ensure_distance_matrix() is None  # should not raise a warning since the shape matches

    dda.distance_matrix = np.array([1.0])  # incorrect shape
    with pytest.warns(UserWarning, match="Distance matrix shape does not match dataset; recomputing."):
        dda._ensure_distance_matrix()
    
    assert dda.distance_matrix.shape[0] == dda.dataset_array.shape[0]  # should have recomputed the distance matrix to match the dataset shape

def test_pre_dda_processing(distance_matrix_dda):
    dda = distance_matrix_dda
    dda.pre_dda_processing()
    assert dda.distance_matrix.shape[0] == dda.dataset_array.shape[0]  # should have computed the distance matrix to match the dataset shape

def test_add_input_vector_to_dda(distance_matrix_dda):
    dda = distance_matrix_dda
    dda.compute_distance_matrix(dda.dataset_array)
    initial_dataset_size = dda.dataset_array.shape[0]
    initial_distance_matrix_size = dda.distance_matrix.shape[0]
    dda.add_input_vector_to_dda()
    assert dda.dataset_array.shape[0] == initial_dataset_size + 1
    assert np.array_equal(dda.dataset_array[-1], dda.input_vector)
    assert dda.distance_matrix.shape[0] == initial_distance_matrix_size + 1