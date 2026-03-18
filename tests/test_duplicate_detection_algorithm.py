import pytest
import numpy as np
from deduplicate.core.duplicate_detection_algorithm import DuplicateDetectionAlgorithm


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        DuplicateDetectionAlgorithm()


def test_concrete_implementation(dummy_dda):
    duplicate = dummy_dda.duplicate_check()
    assert isinstance(duplicate, bool)
    assert duplicate is True

    duplicate_count = dummy_dda.get_dataset_unique_structures()
    assert isinstance(duplicate_count, int)
    assert duplicate_count == 1


def test_calculate_distance(dummy_dda):
    permanent_vector = np.array([1.0, 2.0])
    variable_vector = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [10.0, 20.0],
            [-1.0, -2.0],
        ]
    )
    expected_distance_results_dict = {
        "euclidean": [
            np.linalg.norm(permanent_vector - variable_vector[i])
            for i in range(variable_vector.shape[0])
        ],
        "manhattan": [
            np.sum(np.abs(permanent_vector - variable_vector[i]))
            for i in range(variable_vector.shape[0])
        ],
        "hamming": [
            np.sum(permanent_vector != variable_vector[i])
            for i in range(variable_vector.shape[0])
        ],
        "cosine": [
            1
            - np.dot(permanent_vector, variable_vector[i])
            / (np.linalg.norm(permanent_vector) * np.linalg.norm(variable_vector[i]))
            for i in range(variable_vector.shape[0])
        ],
    }

    tolerance = 1e-5
    for distance_metric, expected_results in expected_distance_results_dict.items():
        dummy_dda.distance_metric = distance_metric
        results = [
            dummy_dda.calculate_distance(permanent_vector, variable_vector[i])
            for i in range(variable_vector.shape[0])
        ]
        assert np.allclose(results, expected_results, atol=tolerance)

def test_compute_distance_matrix(dummy_dda):
    dummy_dda.compute_distance_matrix()
    assert dummy_dda.distance_matrix.shape == (1, 1)
    assert np.isclose(dummy_dda.distance_matrix[0, 0], 0.0)

def test_get_new_distance_matrix_column(dummy_dda):
    new_distances = dummy_dda.get_new_distance_matrix_column()
    assert new_distances.shape == (1,)
    assert np.isclose(new_distances[0], 0.0)

def test_add_new_vector_to_distance_matrix(dummy_dda):
    dummy_dda.compute_distance_matrix()
    dummy_dda.add_new_vector_to_distance_matrix()
    assert dummy_dda.distance_matrix.shape == (2, 2)
