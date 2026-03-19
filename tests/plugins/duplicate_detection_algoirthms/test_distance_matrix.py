from deduplicate.plugins.duplicate_detection_algorithms.distance_matrix import (
    DistanceMatrix,
)
import numpy as np
import pytest

def test_duplicate_check():
    dda = DistanceMatrix(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
        distance_matrix=np.array([]),  # will be computed
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
    dda = DistanceMatrix(
        tolerance=0.1,
        input_vector=np.array([0.0, 0.0]), # not used here
        dataset_array=np.array([[1.0, 2.0], [1.01, 2.01], [10.0, 20.0]]),
        distance_matrix=np.array([]),  # will be computed
        distance_metric="euclidean",
    )
    unique_count = dda.get_dataset_unique_structures()
    assert unique_count == 2
