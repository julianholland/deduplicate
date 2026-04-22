import pytest
import numpy as np
from deduplicate_lib.core.duplicate_detection_algorithm import DuplicateDetectionAlgorithm
import re
from deduplicate_lib.core.duplicate_detection_algorithm import fast_compute_distance_matrix, euclidean_distance, manhattan_distance, hamming_distance, cosine_distance


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        DuplicateDetectionAlgorithm() # type: ignore


def test_concrete_implementation(dummy_dda):
    duplicate = dummy_dda.duplicate_check()
    assert isinstance(duplicate, bool)
    assert duplicate is True

    duplicate_count = dummy_dda.get_dataset_unique_structures()
    assert isinstance(duplicate_count, int)
    assert duplicate_count == 1

def test_distance_metric_setter(dummy_dda):
    dummy_dda.distance_metric = "manhattan"
    assert dummy_dda.distance_metric == "manhattan"

def test_distance_metric_setter_invalid(dummy_dda):
    with pytest.raises(ValueError, match=re.escape("Unsupported distance metric: invalid_metric, supported metrics are: ['euclidean', 'manhattan', 'cosine', 'hamming']")):
        dummy_dda.distance_metric = "invalid_metric"

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
        print(f"Results: {results}")
        print(f"Expected: {expected_results}")
        assert np.allclose(results, expected_results, atol=tolerance)

def test_compute_distance_matrix(dummy_dda):
    dummy_dda.compute_distance_matrix(dummy_dda.dataset_array)
    assert dummy_dda.distance_matrix.shape == (dummy_dda.max_vector_array_size, dummy_dda.max_vector_array_size)
    assert dummy_dda.vector_count == 1
    assert np.isclose(dummy_dda.distance_matrix[0, 0], 0.0)

def test_get_new_distance_matrix_column(dummy_dda):
    new_distances = dummy_dda.get_new_distance_matrix_column(dummy_dda.dataset_array)
    assert new_distances.shape == (1,)
    assert np.isclose(new_distances[0], 0.0)

def test_add_new_vector_to_distance_matrix(dummy_dda):
    dummy_dda.compute_distance_matrix(dummy_dda.dataset_array)
    dummy_dda.add_new_vector_to_distance_matrix(dummy_dda.dataset_array)
    assert dummy_dda.distance_matrix.shape == (dummy_dda.max_vector_array_size, dummy_dda.max_vector_array_size)
    assert dummy_dda.vector_count == 1 # since the dataset has only one vector, adding it again should not change the vector count

def test_pass_only_methods_do_not_raise(dummy_dda):
    dummy_dda.pre_dda_processing()
    dummy_dda.add_input_vector_to_dda()

def test_get_unique_vector_indices_with_mismatch(dummy_dda):
    dummy_dda.compute_distance_matrix(dummy_dda.dataset_array)
    with pytest.raises(ValueError, match=re.escape("Unique vector indices array shape does not match dataset; please run get_dataset_unique_structures() to update the unique vector indices before calling this method.")):
        dummy_dda.get_unique_vector_indices()

def test_get_unique_vector_indices(dummy_dda):
    unique_indices = np.array([True])  # since the dataset has only one vector, it is unique
    dummy_dda.unique_vector_indices = unique_indices
    dummy_dda.get_unique_vector_indices()  # should not raise an error since the shape matches
    assert dummy_dda.unique_vector_indices.shape == (1,)
    assert dummy_dda.unique_vector_indices[0] is np.bool_(True)


def test_fast_compute_distance_matrix(dummy_dda):
    dummy_dda.set_dataset_array(np.array([[1.0, 2.0], [1.0, 2.0], [10.0, 20.0]]))
    dm=fast_compute_distance_matrix(dummy_dda.get_filled_dataset_array(), euclidean_distance)
    assert dm.shape == (3, 3)
    assert np.isclose(dm[0, 0], 0.0)
    assert np.isclose(dm[0, 1], 0.0)
    assert np.isclose(dm[0, 2], np.linalg.norm(np.array([1.0, 2.0]) - np.array([10.0, 20.0])))
    dummy_dda.set_dataset_array(np.array([[1.0, 2.0]]))

def test_fast_euclidean_distance():
    vector1 = np.array([1.0, 2.0])
    vector2 = np.array([4.0, 6.0])
    expected_distance = np.linalg.norm(vector1 - vector2)
    computed_distance = euclidean_distance(vector1, vector2)
    assert np.isclose(computed_distance, expected_distance)

def test_fast_manhattan_distance():
    vector1 = np.array([1.0, 2.0])
    vector2 = np.array([4.0, 6.0])
    expected_distance = np.sum(np.abs(vector1 - vector2))
    computed_distance = manhattan_distance(vector1, vector2)
    assert np.isclose(computed_distance, expected_distance)

def test_fast_hamming_distance():
    vector1 = np.array([1.0, 2.0])
    vector2 = np.array([1.0, 3.0])
    expected_distance = np.sum(vector1 != vector2)
    computed_distance = hamming_distance(vector1, vector2)
    assert np.isclose(computed_distance, expected_distance)

def test_fast_cosine_distance():
    vector1 = np.array([1.0, 2.0])
    vector2 = np.array([4.0, 6.0])
    expected_distance = 1 - np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    computed_distance = cosine_distance(vector1, vector2)
    assert np.isclose(computed_distance, expected_distance)

def test_unimplemented_duplicate_check(dummy_dda):
    class BadDummyDDA(DuplicateDetectionAlgorithm):
        pass
    with pytest.raises(TypeError):
        BadDummyDDA(
            tolerance=0.1,
            input_vector=np.array([1.0, 2.0]),
            dataset_array=np.array([[1.0, 2.0]]),
        ) # type: ignore 

def test_deduplicate(dummy_dda):
    dda=dummy_dda
    dda.set_dataset_array(np.array([[1.0, 2.0], [1.0, 2.0], [10.0, 20.0]]))
    dda.vector_count=3
    dda.unique_vector_indices = np.array([True, False, True])  
    deduplicated_dataset = dda.deduplicate()
    assert deduplicated_dataset.shape == (2, 2)
    assert np.array_equal(deduplicated_dataset, np.array([[1.0, 2.0], [10.0, 20.0]]))
    dda.set_dataset_array(np.array([[1.0, 2.0]]))

    
    