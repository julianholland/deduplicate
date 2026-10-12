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

def test_duplicate_check_does_not_build_distance_matrix():
    dda = DistanceMatrix(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
        distance_matrix=np.array([]),
        distance_metric="euclidean",
    )
    dda.duplicate_check()
    assert dda.distance_matrix.size == 0
    assert dda._dirty is True

    dda.get_dataset_unique_structures()
    assert dda.distance_matrix.size > 0
    assert dda._dirty is False


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

def test_pre_dda_processing(distance_matrix_dda):
    dda = distance_matrix_dda
    dda.pre_dda_processing()
    assert dda.distance_matrix.shape[0] == dda.dataset_array.shape[0]  # should have computed the distance matrix to match the dataset shape

def test_add_input_vector_to_dda(distance_matrix_dda):
    dda = distance_matrix_dda
    dda.compute_distance_matrix(dda.dataset_array)
    initial_dataset_size = dda.get_filled_dataset_array().shape[0]
    initial_distance_matrix_size = dda.get_filled_distance_matrix().shape[0]
    dda.add_input_vector_to_dda()
    assert dda.get_filled_dataset_array().shape[0] == initial_dataset_size + 1
    assert np.array_equal(dda.get_filled_dataset_array()[-1], dda.input_vector)
    assert dda.get_filled_distance_matrix().shape[0] == initial_distance_matrix_size + 1


def test_add_input_vector_to_dda_without_prior_pre_dda_processing():
    dda = DistanceMatrix(
        tolerance=0.1,
        input_vector=np.array([10.0, 20.0]),
        dataset_array=np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
    )
    initial_dataset_size = dda.get_filled_dataset_array().shape[0]
    dda.add_input_vector_to_dda()
    assert dda.get_filled_dataset_array().shape[0] == initial_dataset_size + 1
    assert np.array_equal(dda.get_filled_dataset_array()[-1], dda.input_vector)
    assert dda.get_filled_distance_matrix().shape[0] == initial_dataset_size + 1

def _unique_indices(data, tolerance):
    data = np.asarray(data, dtype=float)
    dda = DistanceMatrix(dataset_array=data, max_vector_array_size=len(data))
    dda.tolerance = tolerance
    count = dda.get_dataset_unique_structures()
    indices = [int(i) for i in dda.get_unique_vector_indices()]
    assert count == len(indices)
    return indices


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        # A group that doesn't contain row 0 keeps its first member.
        ([[5.0], [0.0], [0.1]], [0, 1]),
        # Several groups: one vector each, always the earliest.
        ([[5.0], [0.0], [0.1], [0.2], [9.0], [5.1], [9.2]], [0, 1, 4]),
        # A chain: 0.8 is only compared with kept vectors (0.0), not with
        # the flagged 0.4 between them.
        ([[0.0], [0.4], [0.8]], [0, 2]),
        # Everything within tolerance of the first vector.
        ([[0.0], [0.1], [0.2], [0.3]], [0]),
        # Nothing within tolerance of anything.
        ([[0.0], [1.0], [2.0], [3.0]], [0, 1, 2, 3]),
    ],
)
def test_one_vector_kept_per_group(data, expected):
    assert _unique_indices(data, tolerance=0.5) == expected


def test_deduplicate_returns_one_vector_per_group():
    data = np.array([[5.0, 0.0], [0.0, 0.0], [0.1, 0.0], [5.1, 0.0]])
    dda = DistanceMatrix(dataset_array=data, max_vector_array_size=len(data))
    dda.tolerance = 0.5
    dda.get_dataset_unique_structures()
    assert np.array_equal(dda.deduplicate(), data[[0, 1]])


def test_agrees_with_multi_hashing_on_separated_clusters():
    """Well-separated clusters: both algorithms keep each cluster's first
    vector."""
    from deduplicate_lib.plugins.duplicate_detection_algorithms.multi_hashing import (
        MultiHashing,
    )

    rng = np.random.default_rng(803)
    centres = rng.uniform(0, 10, size=(6, 4))
    data = np.vstack(
        [centre + rng.normal(0, 1e-4, size=(5, 4)) for centre in centres]
    )
    order = rng.permutation(len(data))
    data = data[order]
    first_of_each_cluster = sorted(
        int(np.where(order // 5 == c)[0].min()) for c in range(6)
    )

    assert _unique_indices(data, tolerance=0.5) == first_of_each_cluster

    mh = MultiHashing(
        dataset_array=data, max_vector_array_size=len(data), tolerance=0.5
    )
    mh.get_dataset_unique_structures()
    assert [int(i) for i in mh.get_unique_vector_indices()] == first_of_each_cluster
