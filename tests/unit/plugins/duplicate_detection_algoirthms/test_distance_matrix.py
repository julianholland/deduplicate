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
    print(dda.distance_matrix)
    assert unique_count == 2
