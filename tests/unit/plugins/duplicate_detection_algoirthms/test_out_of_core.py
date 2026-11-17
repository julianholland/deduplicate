"""MultiHashing / DistanceMatrix over out-of-core datasets (VectorStores)."""

import logging

import h5py
import numba
import numpy as np
import pytest

from deduplicate_lib.core import resources
from deduplicate_lib.core.vector_store import (
    CallableStore,
    HDF5Store,
    InMemoryStore,
    NpyStore,
)
from deduplicate_lib.plugins.duplicate_detection_algorithms.distance_matrix import (
    DistanceMatrix,
)
from deduplicate_lib.plugins.duplicate_detection_algorithms.multi_hashing import (
    MultiHashing,
)

from .reference_multi_hashing import reference_unique_mask
from .test_multi_hashing import clustered_dataset

DATA = clustered_dataset(seed=7, n_clusters=40, per_cluster=5, dim=6)
MH_KWARGS = {"tolerance": 0.1, "perturbations": 30, "seed": 5}


def dataset_for(kind, tmp_path, data=DATA):
    if kind == "npy":
        np.save(tmp_path / "data.npy", data)
        return tmp_path / "data.npy"
    if kind == "hdf5":
        with h5py.File(tmp_path / "data.h5", "w") as h5:
            h5.create_dataset("vectors", data=data, maxshape=(None, data.shape[1]))
        return tmp_path / "data.h5"
    if kind == "callable":
        return CallableStore(len(data), data.shape[1], lambda a, b: data[a:b])
    raise ValueError(kind)


OUT_OF_CORE = ["npy", "hdf5", "callable"]


@pytest.fixture
def in_memory_mask():
    dda = MultiHashing(dataset_array=DATA, **MH_KWARGS)
    dda.get_dataset_unique_structures()
    return dda.get_unique_vector_mask()


@pytest.mark.parametrize("kind", OUT_OF_CORE)
def test_out_of_core_mask_matches_in_memory(kind, tmp_path, in_memory_mask):
    dda = MultiHashing(dataset=dataset_for(kind, tmp_path), **MH_KWARGS)
    assert dda.is_out_of_core
    assert dda.vector_count == len(DATA)
    assert dda.get_dataset_unique_structures() == in_memory_mask.sum()
    assert np.array_equal(dda.get_unique_vector_mask(), in_memory_mask)
    expected = reference_unique_mask(
        DATA, dda.perturbation_array, 0.1, dda.acceptance_threshold
    )
    assert np.array_equal(in_memory_mask, expected)


@pytest.mark.parametrize("kind", OUT_OF_CORE)
def test_tiny_chunks_and_batches_match(kind, tmp_path, in_memory_mask, monkeypatch):
    monkeypatch.setattr(resources, "chunk_rows", lambda *_: 7)
    monkeypatch.setattr(resources, "perturbation_batch", lambda *_: 4)
    dda = MultiHashing(dataset=dataset_for(kind, tmp_path), **MH_KWARGS)
    dda.get_dataset_unique_structures()
    assert np.array_equal(dda.get_unique_vector_mask(), in_memory_mask)


def test_memory_limit_drives_chunking(tmp_path, in_memory_mask, caplog):
    """A memory limit far below the dataset size still gives the same answer."""
    # 6 floats x 8 bytes per row: chunks of 4 rows; hashes of 200 vectors: 1/batch
    limit = 4 * 6 * 8 * 4
    dda = MultiHashing(
        dataset=dataset_for("npy", tmp_path), memory_limit=limit, **MH_KWARGS
    )
    with caplog.at_level(logging.INFO, logger="deduplicate_lib"):
        dda.get_dataset_unique_structures()
    assert np.array_equal(dda.get_unique_vector_mask(), in_memory_mask)
    assert "30 batch(es)" in caplog.text
    assert "chunk(s) of up to 4 rows" in caplog.text


def test_hash_tables_dropped_when_too_large(tmp_path, in_memory_mask, caplog):
    dda = MultiHashing(
        dataset=dataset_for("npy", tmp_path), memory_limit="10KB", **MH_KWARGS
    )
    with caplog.at_level(logging.WARNING):
        dda.get_dataset_unique_structures()
    assert "keeping only the unique counts" in caplog.text
    assert dda.sorted_hashes is None
    assert np.array_equal(dda.get_unique_vector_mask(), in_memory_mask)

    dda.input_vector = DATA[0]
    with pytest.raises(MemoryError, match="index_dir"):
        dda.duplicate_check()
    with pytest.raises(MemoryError, match="get_uniqueness_score"):
        dda.get_uniqueness_score()
    with pytest.raises(MemoryError, match="Adding vectors"):
        dda.add_input_vector_to_dda()


@pytest.mark.parametrize("kind", OUT_OF_CORE)
def test_duplicate_check_out_of_core(kind, tmp_path):
    dda = MultiHashing(
        dataset=dataset_for(kind, tmp_path), input_vector=DATA[3], **MH_KWARGS
    )
    assert dda.duplicate_check()
    dda.input_vector = np.full(6, 1e3)
    assert not dda.duplicate_check()
    assert dda.get_uniqueness_score() == 1.0


def test_append_to_hdf5_store(tmp_path):
    path = dataset_for("hdf5", tmp_path, DATA[:150])
    dda = MultiHashing(dataset=path, **MH_KWARGS)
    for vector in DATA[150:]:
        dda.input_vector = vector
        dda.add_input_vector_to_dda()
    assert dda.vector_count == len(DATA) == len(HDF5Store(path))
    dda.get_dataset_unique_structures()

    batch = MultiHashing(dataset_array=DATA, **MH_KWARGS)
    batch.get_dataset_unique_structures()
    assert np.array_equal(dda.get_unique_vector_mask(), batch.get_unique_vector_mask())


def test_append_grows_unique_counts_past_preallocation(tmp_path):
    path = dataset_for("hdf5", tmp_path, DATA[:10])
    dda = MultiHashing(dataset=path, max_vector_array_size=10, **MH_KWARGS)
    for vector in DATA[10:30]:
        dda.input_vector = vector
        dda.add_input_vector_to_dda()
    assert dda.vector_count == 30
    assert dda.unique_counts.shape[0] >= 30


@pytest.mark.parametrize("kind", ["npy", "callable"])
def test_append_to_read_only_store_raises(kind, tmp_path):
    dda = MultiHashing(
        dataset=dataset_for(kind, tmp_path), input_vector=DATA[0], **MH_KWARGS
    )
    with pytest.raises(NotImplementedError, match="read-only"):
        dda.add_input_vector_to_dda()


def test_out_of_core_dataset_array_raises(tmp_path):
    dda = MultiHashing(dataset=dataset_for("npy", tmp_path), **MH_KWARGS)
    with pytest.raises(ValueError, match="out-of-core"):
        _ = dda.dataset_array
    with pytest.raises(ValueError, match="out-of-core"):
        dda.get_filled_dataset_array()


def test_input_vector_length_checked_against_store(tmp_path):
    dda = MultiHashing(
        dataset=dataset_for("npy", tmp_path), input_vector=np.zeros(3), **MH_KWARGS
    )
    with pytest.raises(ValueError, match="Input vector length"):
        dda.duplicate_check()


def test_dataset_and_dataset_array_are_exclusive(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        MultiHashing(
            dataset=dataset_for("npy", tmp_path), dataset_array=DATA, **MH_KWARGS
        )


def test_dataset_accepts_arrays_and_in_memory_stores(in_memory_mask):
    for dataset in (DATA, InMemoryStore(DATA)):
        dda = MultiHashing(dataset=dataset, **MH_KWARGS)
        assert not dda.is_out_of_core
        dda.get_dataset_unique_structures()
        assert np.array_equal(dda.get_unique_vector_mask(), in_memory_mask)


def test_set_dataset_array_replaces_store(tmp_path, in_memory_mask):
    dda = MultiHashing(dataset=dataset_for("npy", tmp_path, DATA[:20]), **MH_KWARGS)
    dda.get_dataset_unique_structures()
    dda.set_dataset_array(DATA)
    assert not dda.is_out_of_core
    dda.get_dataset_unique_structures()
    assert np.array_equal(dda.get_unique_vector_mask(), in_memory_mask)


def test_store_property_for_in_memory_dataset():
    dda = MultiHashing(dataset_array=DATA, **MH_KWARGS)
    assert isinstance(dda.store, InMemoryStore)
    assert dda.store.shape == DATA.shape  # filled rows only, not the preallocation
    empty = MultiHashing(input_vector=np.zeros(6), **MH_KWARGS)
    assert empty.store.shape == (0, 6)


def test_n_threads_does_not_change_results(tmp_path, in_memory_mask):
    before = numba.get_num_threads()
    try:
        for n_threads in (1, 2):
            dda = MultiHashing(
                dataset=dataset_for("npy", tmp_path), n_threads=n_threads, **MH_KWARGS
            )
            dda.get_dataset_unique_structures()
            assert np.array_equal(dda.get_unique_vector_mask(), in_memory_mask)
    finally:
        numba.set_num_threads(before)
    with pytest.raises(ValueError, match="n_threads"):
        MultiHashing(n_threads=0, **MH_KWARGS)


def test_progress_bars(tmp_path, in_memory_mask, capsys):
    dda = MultiHashing(dataset=dataset_for("npy", tmp_path), progress=True, **MH_KWARGS)
    dda.get_dataset_unique_structures()
    assert np.array_equal(dda.get_unique_vector_mask(), in_memory_mask)
    assert "hashing (batch 1/1)" in capsys.readouterr().err


# --- deduplicate() ---


def test_deduplicate_in_memory_returns_array(in_memory_mask):
    dda = MultiHashing(dataset_array=DATA, **MH_KWARGS)
    result = dda.deduplicate()  # computes the mask if there is none yet
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, DATA[in_memory_mask])


def test_deduplicate_in_memory_over_limit_warns_and_returns(caplog):
    dda = MultiHashing(dataset_array=DATA, memory_limit=100, **MH_KWARGS)
    with caplog.at_level(logging.WARNING):
        result = dda.deduplicate()
    assert isinstance(result, np.ndarray)
    assert "deduplicate(out=...)" in caplog.text


@pytest.mark.parametrize("suffix", [".npy", ".h5"])
def test_deduplicate_in_memory_with_out_writes_file(suffix, tmp_path, in_memory_mask):
    dda = MultiHashing(dataset_array=DATA, **MH_KWARGS)
    written = dda.deduplicate(out=tmp_path / f"unique{suffix}")
    assert written.path == tmp_path / f"unique{suffix}"
    assert np.array_equal(written.read(0, len(written)), DATA[in_memory_mask])


@pytest.mark.parametrize(
    ("kind", "store_type"), [("npy", NpyStore), ("hdf5", HDF5Store)]
)
def test_deduplicate_out_of_core_mirrors_input_format(
    kind, store_type, tmp_path, in_memory_mask, caplog
):
    path = dataset_for(kind, tmp_path)
    dda = MultiHashing(dataset=path, memory_limit="1KB", **MH_KWARGS)
    with caplog.at_level(logging.INFO, logger="deduplicate_lib"):
        written = dda.deduplicate()
    assert isinstance(written, store_type)
    assert written.path == path.with_name(f"data_unique{path.suffix}")
    assert np.array_equal(written.read(0, len(written)), DATA[in_memory_mask])
    assert f"Wrote {in_memory_mask.sum()} unique vectors" in caplog.text


def test_deduplicate_callable_needs_out(tmp_path, in_memory_mask):
    dda = MultiHashing(dataset=dataset_for("callable", tmp_path), **MH_KWARGS)
    with pytest.raises(ValueError, match=r"deduplicate\(out=...\)"):
        dda.deduplicate()
    written = dda.deduplicate(out=tmp_path / "unique.h5")
    assert np.array_equal(written.read(0, len(written)), DATA[in_memory_mask])


# --- DistanceMatrix refuses what it cannot hold ---


@pytest.mark.parametrize("kind", OUT_OF_CORE)
def test_distance_matrix_refuses_out_of_core(kind, tmp_path):
    with pytest.raises(ValueError, match="Use MultiHashing"):
        DistanceMatrix(dataset=dataset_for(kind, tmp_path))
    dm = DistanceMatrix(dataset_array=DATA[:5], max_vector_array_size=10)
    with pytest.raises(ValueError, match="Use MultiHashing"):
        dm.set_dataset_store(dataset_for(kind, tmp_path))


def test_distance_matrix_refuses_matrix_over_memory_limit():
    with pytest.raises(MemoryError, match="max_vector_array_size"):
        DistanceMatrix(
            dataset_array=DATA, max_vector_array_size=1000, memory_limit="1MB"
        )
    DistanceMatrix(dataset_array=DATA, max_vector_array_size=300, memory_limit="1MB")


def test_distance_matrix_accepts_in_memory_dataset_argument():
    dm = DistanceMatrix(dataset=DATA, max_vector_array_size=len(DATA), tolerance=0.5)
    assert dm.get_dataset_unique_structures() > 0


# --- deduplicate(out=<the input file>) must not destroy the input ---


@pytest.mark.parametrize("kind", ["npy", "hdf5"])
def test_deduplicate_refuses_to_overwrite_input(kind, tmp_path):
    path = dataset_for(kind, tmp_path)
    before = path.read_bytes()
    dda = MultiHashing(dataset=path, **MH_KWARGS)
    with pytest.raises(ValueError, match="Refusing"):
        dda.deduplicate(out=path)
    assert path.read_bytes() == before
    assert np.array_equal(dda.store.read(0, len(DATA)), DATA)


# --- add_vectors keeps vector_count aligned with the tables when it fails ---


def failing_on_second_call(dda, monkeypatch):
    """Force 5-row chunks and make the second ``_add_rows_to_tables`` call fail."""
    original = dda._add_rows_to_tables
    calls = []

    def flaky(rows, start):
        calls.append(start)
        if len(calls) == 2:
            raise RuntimeError("simulated failure")
        return original(rows, start)

    monkeypatch.setattr(dda, "_add_chunk_rows", lambda: 5)
    monkeypatch.setattr(dda, "_add_rows_to_tables", flaky)
    return original


def test_add_vectors_failure_keeps_count_aligned_in_memory(monkeypatch):
    dda = MultiHashing(dataset_array=DATA[:100], max_vector_array_size=300, **MH_KWARGS)
    dda.get_dataset_unique_structures()
    failing_on_second_call(dda, monkeypatch)
    with pytest.raises(RuntimeError, match="simulated"):
        dda.add_vectors(DATA[100:130])
    # first chunk of 5 rows was hashed, the second was not
    assert dda.vector_count == dda._rows_in_tables == 105

    fresh = MultiHashing(dataset_array=DATA[:105], **MH_KWARGS)
    fresh.get_dataset_unique_structures()
    dda.get_dataset_unique_structures()
    assert np.array_equal(dda.get_unique_vector_mask(), fresh.get_unique_vector_mask())


def test_add_vectors_recovers_after_partial_failure_hdf5(tmp_path, monkeypatch):
    path = dataset_for("hdf5", tmp_path, DATA[:100])
    dda = MultiHashing(dataset=path, **MH_KWARGS)
    dda.get_dataset_unique_structures()
    with monkeypatch.context() as patch:
        failing_on_second_call(dda, patch)
        with pytest.raises(RuntimeError, match="simulated"):
            dda.add_vectors(DATA[100:130])
    assert dda.vector_count == dda._rows_in_tables == 105
    # the store already holds all 130 rows
    assert len(dda.store) == 130

    dda.add_vectors(DATA[130:150])
    assert dda.vector_count == 150 == len(dda.store)
    dda.get_dataset_unique_structures()

    fresh = MultiHashing(dataset_array=DATA[:150], **MH_KWARGS)
    fresh.get_dataset_unique_structures()
    assert np.array_equal(dda.get_unique_vector_mask(), fresh.get_unique_vector_mask())
