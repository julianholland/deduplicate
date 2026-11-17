"""``DuplicateDetectionAlgorithm.tuning_copy``: an independent in-memory DDA for tuning."""

import numpy as np

from deduplicate_lib.plugins.duplicate_detection_algorithms.distance_matrix import (
    DistanceMatrix,
)
from deduplicate_lib.plugins.duplicate_detection_algorithms.multi_hashing import (
    MultiHashing,
)

from .test_multi_hashing import clustered_dataset

DATA = clustered_dataset(seed=3, n_clusters=10, per_cluster=4, dim=5)
OTHER = clustered_dataset(seed=4, n_clusters=6, per_cluster=3, dim=5)


def mh_kwargs():
    return {
        "tolerance": 0.2,
        "perturbations": 17,
        "seed": 9,
        "sigma_accepatnce_threshold": 2,
    }


def test_distance_matrix_tuning_copy():
    dda = DistanceMatrix(dataset_array=DATA, tolerance=0.3, distance_metric="hamming")
    copy = dda.tuning_copy(OTHER)
    assert type(copy) is DistanceMatrix
    assert copy is not dda
    assert copy.tolerance == 0.3
    assert copy.distance_metric == dda.distance_metric
    assert copy.vector_count == len(OTHER)
    assert copy.max_vector_array_size == len(OTHER)
    assert not copy.is_out_of_core
    assert np.array_equal(copy.get_filled_dataset_array(), OTHER)


def test_distance_matrix_tuning_copy_is_independent():
    dda = DistanceMatrix(dataset_array=DATA, tolerance=0.3)
    dda.pre_dda_processing()
    before = dda.get_filled_dataset_array().copy()
    copy = dda.tuning_copy(OTHER)
    copy.tolerance = 5.0
    copy.get_dataset_unique_structures()
    copy.set_dataset_array(OTHER[:5])
    assert dda.tolerance == 0.3
    assert dda.vector_count == len(DATA)
    assert np.array_equal(dda.get_filled_dataset_array(), before)


def test_multi_hashing_tuning_copy_keeps_settings():
    dda = MultiHashing(dataset_array=DATA, **mh_kwargs())
    copy = dda.tuning_copy(OTHER)
    assert type(copy) is MultiHashing
    assert copy.tolerance == 0.2
    assert copy.perturbations == 17
    assert copy.seed == 9
    assert copy.sigma_accepatnce_threshold == 2
    assert copy.distance_metric == dda.distance_metric
    assert copy.vector_count == len(OTHER)
    assert np.array_equal(copy.perturbation_array, dda.perturbation_array)
    assert not copy.is_out_of_core


def test_multi_hashing_tuning_copy_matches_fresh_dda():
    dda = MultiHashing(dataset_array=DATA, **mh_kwargs())
    copy = dda.tuning_copy(OTHER)
    fresh = MultiHashing(dataset_array=OTHER, **mh_kwargs())
    assert copy.get_dataset_unique_structures() == fresh.get_dataset_unique_structures()
    assert np.array_equal(copy.get_unique_vector_mask(), fresh.get_unique_vector_mask())


def test_multi_hashing_tuning_copy_is_independent():
    dda = MultiHashing(dataset_array=DATA, **mh_kwargs())
    dda.get_dataset_unique_structures()
    unique_before = dda.get_dataset_unique_structures()
    before = dda.get_filled_dataset_array().copy()
    copy = dda.tuning_copy(OTHER)
    copy.tolerance = 50.0
    copy.get_dataset_unique_structures()
    copy.perturbation_array[:] = 0.0
    assert dda.tolerance == 0.2
    assert not np.all(dda.perturbation_array == 0.0)
    assert np.array_equal(dda.get_filled_dataset_array(), before)
    assert dda.get_dataset_unique_structures() == unique_before


def test_tuning_copy_of_out_of_core_multi_hashing(tmp_path):
    np.save(tmp_path / "data.npy", DATA)
    dda = MultiHashing(dataset=tmp_path / "data.npy", **mh_kwargs())
    assert dda.is_out_of_core
    copy = dda.tuning_copy(OTHER)
    assert type(copy) is MultiHashing
    assert not copy.is_out_of_core
    assert copy.vector_count == len(OTHER)
    assert copy.perturbations == 17
    assert copy.get_dataset_unique_structures() > 0
    # the original is still out-of-core and unchanged
    assert dda.is_out_of_core
    assert dda.vector_count == len(DATA)


def test_tuning_copy_of_indexed_multi_hashing_has_no_index_dir(tmp_path):
    np.save(tmp_path / "data.npy", DATA)
    index_dir = tmp_path / "index"
    dda = MultiHashing(
        dataset=tmp_path / "data.npy", index_dir=index_dir, **mh_kwargs()
    )
    dda.get_dataset_unique_structures()
    metadata = (index_dir / "metadata.json").read_bytes()
    files = sorted(p.name for p in index_dir.iterdir())

    copy = dda.tuning_copy(OTHER)
    assert copy.index_dir is None
    assert not copy.is_out_of_core
    copy.tolerance = 1.0
    copy.get_dataset_unique_structures()

    assert (index_dir / "metadata.json").read_bytes() == metadata
    assert sorted(p.name for p in index_dir.iterdir()) == files
    assert dda.is_out_of_core


def test_tuning_copy_of_auto_limit_dda_stays_auto():
    dda = MultiHashing(dataset_array=DATA, **mh_kwargs())
    assert dda.memory_limit_is_auto
    assert dda._init_kwargs()["memory_limit"] == "auto"
    assert dda.tuning_copy(OTHER).memory_limit_is_auto


def test_tuning_copy_of_explicit_limit_dda_keeps_the_limit():
    dda = MultiHashing(dataset_array=DATA, memory_limit="64MB", **mh_kwargs())
    copy = dda.tuning_copy(OTHER)
    assert not copy.memory_limit_is_auto
    assert copy.memory_limit == dda.memory_limit


def test_tuning_copy_keeps_custom_perturbation_array():
    custom = np.linspace(0.2, 1.5, 17)
    dda = MultiHashing(dataset_array=DATA, pertrubation_array=custom, **mh_kwargs())
    copy = dda.tuning_copy(OTHER)
    copy.get_dataset_unique_structures()
    assert np.array_equal(copy.perturbation_array, custom)
