"""MultiHashing with a persistent hash index (index_dir)."""

import itertools
import json
import logging

import h5py
import numpy as np
import pytest

from deduplicate_lib.core import hash_index, resources
from deduplicate_lib.core.hash_index import (
    METADATA_FILE,
    HashIndex,
    IndexMismatchError,
    fence_name,
)
from deduplicate_lib.core.sorted_segment import SortedSegment
from deduplicate_lib.core.vector_store import CallableStore, HDF5Store
from deduplicate_lib.plugins.duplicate_detection_algorithms.multi_hashing import (
    MultiHashing,
)

from .reference_multi_hashing import reference_unique_mask
from .test_multi_hashing import clustered_dataset

DATA = clustered_dataset(seed=11, n_clusters=60, per_cluster=5, dim=6)
MH_KWARGS = {"tolerance": 0.1, "perturbations": 30, "seed": 5}


def npy_dataset(tmp_path, data=DATA, name="data.npy"):
    np.save(tmp_path / name, data)
    return tmp_path / name


def hdf5_dataset(tmp_path, data=DATA, name="data.h5"):
    with h5py.File(tmp_path / name, "w") as h5:
        h5.create_dataset("vectors", data=data, maxshape=(None, data.shape[1]))
    return tmp_path / name


def _fresh_mask(data=DATA, **kwargs):
    dda = MultiHashing(dataset_array=data, **{**MH_KWARGS, **kwargs})
    dda.get_dataset_unique_structures()
    return dda.get_unique_vector_mask()


# computed up front: some tests forbid hashing once they reopen an index
FRESH_MASK = _fresh_mask()


def fresh_mask():
    return FRESH_MASK


def no_hashing(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("the dataset was hashed again")

    monkeypatch.setattr(MultiHashing, "_hash_perturbations", fail)


def test_build_writes_index_and_matches_in_memory(tmp_path):
    index_dir = tmp_path / "idx"
    dda = MultiHashing(dataset=npy_dataset(tmp_path), index_dir=index_dir, **MH_KWARGS)
    dda.get_dataset_unique_structures()
    assert np.array_equal(dda.get_unique_vector_mask(), fresh_mask())
    expected = reference_unique_mask(
        DATA, dda.perturbation_array, 0.1, dda.acceptance_threshold
    )
    assert np.array_equal(dda.get_unique_vector_mask(), expected)

    metadata = json.loads((index_dir / METADATA_FILE).read_text())
    assert metadata["n_vectors"] == len(DATA)
    assert metadata["tolerance"] == 0.1
    assert metadata["seed"] == 5
    assert all(metadata["hash_files"])
    files = {f.name for f in index_dir.iterdir()}
    fences = {fence_name(name) for name in metadata["hash_files"]}
    assert files == {
        METADATA_FILE,
        metadata["counts_file"],
        *metadata["hash_files"],
        *fences,
    }
    # the tables are memory-mapped from the index, not held in memory
    assert all(
        isinstance(t, SortedSegment) and isinstance(t.table, np.memmap)
        for t in dda.sorted_hashes
    )


def test_index_keeps_tables_even_when_memory_is_tight(tmp_path):
    dda = MultiHashing(
        dataset=npy_dataset(tmp_path),
        index_dir=tmp_path / "idx",
        memory_limit="10KB",
        input_vector=DATA[0],
        **MH_KWARGS,
    )
    assert dda.duplicate_check()  # no MemoryError: the tables are on disk
    dda.get_dataset_unique_structures()
    assert np.array_equal(dda.get_unique_vector_mask(), fresh_mask())


def test_reopen_reuses_index_without_hashing(tmp_path, monkeypatch):
    path = npy_dataset(tmp_path)
    first = MultiHashing(dataset=path, index_dir=tmp_path / "idx", **MH_KWARGS)
    first.get_dataset_unique_structures()

    no_hashing(monkeypatch)
    reopened = MultiHashing(
        dataset=path, index_dir=tmp_path / "idx", input_vector=DATA[7], **MH_KWARGS
    )
    reopened.get_dataset_unique_structures()
    assert np.array_equal(
        reopened.get_unique_vector_mask(), first.get_unique_vector_mask()
    )
    assert reopened.duplicate_check()
    reopened.input_vector = np.full(6, 1e3)
    assert not reopened.duplicate_check()


def test_reopen_with_other_dtype_and_sigma_reuses_index(tmp_path, monkeypatch):
    """The index depends on the values and hashing parameters only: the same
    data loaded in memory as float64 reuses an index built from a float32 file,
    and the acceptance threshold (sigma) is applied at query time."""
    data = DATA.astype(np.float32)
    path = npy_dataset(tmp_path, data)
    MultiHashing(
        dataset=path, index_dir=tmp_path / "idx", **MH_KWARGS
    ).get_dataset_unique_structures()

    no_hashing(monkeypatch)
    in_memory = MultiHashing(
        dataset_array=data.astype(np.float64),
        index_dir=tmp_path / "idx",
        sigma_accepatnce_threshold=2,
        **MH_KWARGS,
    )
    in_memory.get_dataset_unique_structures()
    expected = reference_unique_mask(
        data.astype(np.float64),
        in_memory.perturbation_array,
        0.1,
        in_memory.acceptance_threshold,
    )
    assert np.array_equal(in_memory.get_unique_vector_mask(), expected)


def test_interrupted_build_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "perturbation_batch", lambda *_: 4)
    path = npy_dataset(tmp_path)
    real_commit = HashIndex.commit
    calls = {"commits": 0}

    def crash_on_third_commit(self, counts, store):
        calls["commits"] += 1
        if calls["commits"] == 3:
            raise KeyboardInterrupt("killed")
        real_commit(self, counts, store)

    monkeypatch.setattr(HashIndex, "commit", crash_on_third_commit)
    dda = MultiHashing(dataset=path, index_dir=tmp_path / "idx", **MH_KWARGS)
    with pytest.raises(KeyboardInterrupt):
        dda.get_dataset_unique_structures()
    monkeypatch.setattr(HashIndex, "commit", real_commit)

    metadata = json.loads((tmp_path / "idx" / METADATA_FILE).read_text())
    assert sum(bool(f) for f in metadata["hash_files"]) == 8  # two batches of 4

    hashed = []
    real_hash = MultiHashing._hash_perturbations

    def record(self, store, n_vectors, perturbation_ids, on_table, on_batch):
        hashed.extend(perturbation_ids)
        real_hash(self, store, n_vectors, perturbation_ids, on_table, on_batch)

    monkeypatch.setattr(MultiHashing, "_hash_perturbations", record)
    resumed = MultiHashing(dataset=path, index_dir=tmp_path / "idx", **MH_KWARGS)
    resumed.get_dataset_unique_structures()
    assert hashed == list(range(8, 30))
    assert np.array_equal(resumed.get_unique_vector_mask(), fresh_mask())
    # the uncommitted tables of the crashed batch were cleaned up
    metadata = json.loads((tmp_path / "idx" / METADATA_FILE).read_text())
    files = {f.name for f in (tmp_path / "idx").iterdir()}
    assert files == {
        METADATA_FILE,
        metadata["counts_file"],
        *metadata["hash_files"],
        *(fence_name(name) for name in metadata["hash_files"] if name),
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"tolerance": 0.2}, "tolerance"),
        ({"seed": 6}, "perturbation_sha256"),
        ({"perturbations": 31}, "perturbations"),
    ],
)
def test_parameter_mismatch_raises(tmp_path, change, message):
    path = npy_dataset(tmp_path)
    MultiHashing(
        dataset=path, index_dir=tmp_path / "idx", **MH_KWARGS
    ).get_dataset_unique_structures()
    dda = MultiHashing(
        dataset=path, index_dir=tmp_path / "idx", **{**MH_KWARGS, **change}
    )
    with pytest.raises(IndexMismatchError, match=message):
        dda.get_dataset_unique_structures()


def test_changing_tolerance_after_build_raises(tmp_path):
    dda = MultiHashing(
        dataset=npy_dataset(tmp_path), index_dir=tmp_path / "idx", **MH_KWARGS
    )
    dda.get_dataset_unique_structures()
    dda.tolerance = 0.3
    with pytest.raises(
        IndexMismatchError, match="each tolerance needs its own index_dir"
    ):
        dda.get_dataset_unique_structures()


def test_other_dataset_raises(tmp_path):
    MultiHashing(
        dataset=npy_dataset(tmp_path), index_dir=tmp_path / "idx", **MH_KWARGS
    ).get_dataset_unique_structures()

    other = npy_dataset(tmp_path, DATA[::-1].copy(), name="other.npy")
    with pytest.raises(IndexMismatchError, match="dataset differs"):
        MultiHashing(
            dataset=other, index_dir=tmp_path / "idx", **MH_KWARGS
        ).get_dataset_unique_structures()

    shorter = npy_dataset(tmp_path, DATA[:10], name="short.npy")
    with pytest.raises(IndexMismatchError, match="has only 10"):
        MultiHashing(
            dataset=shorter, index_dir=tmp_path / "idx", **MH_KWARGS
        ).get_dataset_unique_structures()


def test_refuses_non_index_directory(tmp_path):
    (tmp_path / "idx").mkdir()
    (tmp_path / "idx" / "notes.txt").write_text("mine")
    with pytest.raises(IndexMismatchError, match="not a deduplicate_lib index"):
        MultiHashing(
            dataset=npy_dataset(tmp_path), index_dir=tmp_path / "idx", **MH_KWARGS
        ).get_dataset_unique_structures()
    assert (tmp_path / "idx" / "notes.txt").exists()


def test_orphan_files_are_removed_on_open(tmp_path):
    path = npy_dataset(tmp_path)
    MultiHashing(
        dataset=path, index_dir=tmp_path / "idx", **MH_KWARGS
    ).get_dataset_unique_structures()
    orphan = tmp_path / "idx" / "p0003.g999.npy"
    np.save(orphan, np.arange(3, dtype=np.uint64))
    (tmp_path / "idx" / "metadata.json.tmp").write_text("{")
    MultiHashing(
        dataset=path, index_dir=tmp_path / "idx", **MH_KWARGS
    ).get_dataset_unique_structures()
    assert not orphan.exists()
    assert not (tmp_path / "idx" / "metadata.json.tmp").exists()


# --- appending ---


@pytest.mark.parametrize("batched", [False, True])
def test_appends_flush_and_reopen(tmp_path, monkeypatch, batched):
    path = hdf5_dataset(tmp_path, DATA[:200])
    dda = MultiHashing(dataset=path, index_dir=tmp_path / "idx", **MH_KWARGS)
    dda.get_dataset_unique_structures()
    if batched:
        dda.add_vectors(DATA[200:])
    else:
        for vector in DATA[200:]:
            dda.input_vector = vector
            dda.add_input_vector_to_dda()
    dda.flush()
    dda.get_dataset_unique_structures()
    assert np.array_equal(dda.get_unique_vector_mask(), fresh_mask())

    metadata = json.loads((tmp_path / "idx" / METADATA_FILE).read_text())
    assert metadata["n_vectors"] == len(DATA)

    no_hashing(monkeypatch)
    reopened = MultiHashing(dataset=path, index_dir=tmp_path / "idx", **MH_KWARGS)
    reopened.get_dataset_unique_structures()
    assert np.array_equal(reopened.get_unique_vector_mask(), fresh_mask())


def test_unflushed_appends_are_recovered_on_reopen(tmp_path, caplog):
    path = hdf5_dataset(tmp_path, DATA[:200])
    dda = MultiHashing(dataset=path, index_dir=tmp_path / "idx", **MH_KWARGS)
    dda.get_dataset_unique_structures()
    dda.add_vectors(DATA[200:])  # in the HDF5 file, but never flushed
    del dda

    with caplog.at_level(logging.INFO, logger="deduplicate_lib"):
        reopened = MultiHashing(dataset=path, index_dir=tmp_path / "idx", **MH_KWARGS)
        reopened.get_dataset_unique_structures()
    assert "Adding the 100 vectors appended" in caplog.text
    assert np.array_equal(reopened.get_unique_vector_mask(), fresh_mask())
    metadata = json.loads((tmp_path / "idx" / METADATA_FILE).read_text())
    assert metadata["n_vectors"] == len(DATA)


def test_dataset_grown_outside_the_dda_is_caught_up(tmp_path):
    path = hdf5_dataset(tmp_path, DATA[:150])
    MultiHashing(
        dataset=path, index_dir=tmp_path / "idx", **MH_KWARGS
    ).get_dataset_unique_structures()
    HDF5Store(path).append(DATA[150:])
    grown = MultiHashing(dataset=path, index_dir=tmp_path / "idx", **MH_KWARGS)
    grown.get_dataset_unique_structures()
    assert np.array_equal(grown.get_unique_vector_mask(), fresh_mask())


@pytest.mark.parametrize("batched", [False, True])
def test_automatic_flushes_mid_append_stay_consistent(tmp_path, monkeypatch, batched):
    """Flushes triggered part-way through adding vectors commit exactly the rows
    hashed so far; a crash after them still reopens to the right answer."""
    monkeypatch.setattr(MultiHashing, "_pending_limit", lambda self: 40)
    monkeypatch.setattr(MultiHashing, "_add_chunk_rows", lambda self: 7)
    path = hdf5_dataset(tmp_path, DATA[:100])
    dda = MultiHashing(dataset=path, index_dir=tmp_path / "idx", **MH_KWARGS)
    dda.get_dataset_unique_structures()
    generation = json.loads((tmp_path / "idx" / METADATA_FILE).read_text())[
        "generation"
    ]
    if batched:
        dda.add_vectors(DATA[100:])
    else:
        for vector in DATA[100:]:
            dda.input_vector = vector
            dda.add_input_vector_to_dda()
    metadata = json.loads((tmp_path / "idx" / METADATA_FILE).read_text())
    assert metadata["generation"] > generation  # flushed automatically
    assert 100 < metadata["n_vectors"] <= len(DATA)
    del dda  # "crash": the last appends were never flushed

    reopened = MultiHashing(dataset=path, index_dir=tmp_path / "idx", **MH_KWARGS)
    reopened.get_dataset_unique_structures()
    assert np.array_equal(reopened.get_unique_vector_mask(), fresh_mask())


def test_in_memory_dataset_with_index_and_appends(tmp_path, monkeypatch):
    monkeypatch.setattr(MultiHashing, "_pending_limit", lambda self: 40)
    dda = MultiHashing(
        dataset_array=DATA[:100], index_dir=tmp_path / "idx", **MH_KWARGS
    )
    dda.get_dataset_unique_structures()
    for vector in DATA[100:]:
        dda.input_vector = vector
        dda.add_input_vector_to_dda()
    dda.flush()
    dda.get_dataset_unique_structures()
    assert np.array_equal(dda.get_unique_vector_mask(), fresh_mask())

    no_hashing(monkeypatch)
    reopened = MultiHashing(dataset_array=DATA, index_dir=tmp_path / "idx", **MH_KWARGS)
    reopened.get_dataset_unique_structures()
    assert np.array_equal(reopened.get_unique_vector_mask(), fresh_mask())


def test_streaming_into_a_new_hdf5_store_with_index(tmp_path):
    """Start from nothing: an empty appendable store, deduplicating on the way in."""
    store = HDF5Store.create(tmp_path / "stream.h5", dim=6)
    dda = MultiHashing(dataset=store, index_dir=tmp_path / "idx", **MH_KWARGS)
    kept = []
    for i, vector in enumerate(DATA):
        dda.input_vector = vector
        if not dda.duplicate_check():
            dda.add_input_vector_to_dda()
            kept.append(i)
    dda.flush()
    assert len(HDF5Store(tmp_path / "stream.h5")) == len(kept) == dda.vector_count
    assert len(kept) < len(DATA)
    reopened = MultiHashing(
        dataset=tmp_path / "stream.h5", index_dir=tmp_path / "idx", **MH_KWARGS
    )
    reopened.input_vector = DATA[kept[0]]
    assert reopened.duplicate_check()


# --- add_vectors / flush without an index ---


def test_add_vectors_matches_sequential_appends_without_index():
    sequential = MultiHashing(dataset_array=DATA[:50], **MH_KWARGS)
    for vector in DATA[50:]:
        sequential.input_vector = vector
        sequential.add_input_vector_to_dda()
    batched = MultiHashing(dataset_array=DATA[:50], **MH_KWARGS)
    batched.add_vectors(DATA[50:])
    for dda in (sequential, batched):
        dda.get_dataset_unique_structures()
        assert np.array_equal(dda.get_unique_vector_mask(), fresh_mask())
    assert np.array_equal(
        sequential.unique_counts[: len(DATA)], batched.unique_counts[: len(DATA)]
    )
    batched.flush()  # merges pending hashes in memory
    assert not any(batched._pending_hashes)
    batched.input_vector = DATA[-1]
    assert batched.duplicate_check()


def test_add_vectors_errors():
    dda = MultiHashing(dataset_array=DATA[:5], max_vector_array_size=8, **MH_KWARGS)
    with pytest.raises(ValueError, match="wrong length"):
        dda.add_vectors(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="max_vector_array_size"):
        dda.add_vectors(DATA[5:10])
    dda.add_vectors(DATA[5:8])
    dda.input_vector = DATA[9]
    with pytest.raises(ValueError, match="max_vector_array_size"):
        dda.add_input_vector_to_dda()


def test_durable_commits_with_fsync(tmp_path, monkeypatch):
    from deduplicate_lib.core import hash_index

    monkeypatch.setattr(hash_index, "FSYNC", True)
    dda = MultiHashing(
        dataset=npy_dataset(tmp_path),
        index_dir=tmp_path / "idx",
        tolerance=0.1,
        perturbations=3,
    )
    dda.get_dataset_unique_structures()
    assert (
        json.loads((tmp_path / "idx" / METADATA_FILE).read_text())["n_vectors"] == 300
    )


def test_load_table_of_missing_perturbation(tmp_path):
    from deduplicate_lib.core.vector_store import InMemoryStore

    index = HashIndex.open(
        tmp_path / "idx",
        {
            "hash_version": 1,
            "tolerance": 0.1,
            "perturbations": 2,
            "perturbation_sha256": "x",
            "seed": 1,
            "dim": 6,
        },
        InMemoryStore(DATA),
    )
    assert index.completed == []
    assert index.load_counts().shape == (300,)
    with pytest.raises(KeyError, match="no table yet"):
        index.load_table(0)


# --- the dataset fingerprint covers every row ---


def test_changing_a_middle_row_is_detected(tmp_path):
    data = np.random.default_rng(0).normal(size=(5000, 3))
    path = npy_dataset(tmp_path, data)
    kwargs = {"tolerance": 0.1, "perturbations": 5, "seed": 1}
    MultiHashing(
        dataset=path, index_dir=tmp_path / "idx", **kwargs
    ).get_dataset_unique_structures()

    changed = data.copy()
    changed[2500, 1] += 1.0  # neither in the first nor in the last 1024 rows
    np.save(path, changed)
    with pytest.raises(IndexMismatchError, match="dataset differs"):
        MultiHashing(
            dataset=path, index_dir=tmp_path / "idx", **kwargs
        ).get_dataset_unique_structures()


def recording_store(data, reads):
    def fetch(start, stop):
        reads.append((start, stop))
        return data[start:stop]

    return CallableStore(len(data), data.shape[1], fetch)


def test_fingerprint_blocks(monkeypatch):
    monkeypatch.setattr(hash_index, "FINGERPRINT_BLOCK_ROWS", 10)
    data = np.random.default_rng(1).normal(size=(35, 2))
    fingerprint = hash_index.dataset_fingerprint(recording_store(data, []), 35)
    assert fingerprint["n_vectors"] == 35
    assert fingerprint["block_rows"] == 10
    assert len(fingerprint["blocks"]) == 4

    changed = data.copy()
    changed[15, 0] += 1
    other = hash_index.dataset_fingerprint(recording_store(changed, []), 35)
    assert other["blocks"][0] == fingerprint["blocks"][0]
    assert other["blocks"][1] != fingerprint["blocks"][1]
    assert other["blocks"][2:] == fingerprint["blocks"][2:]


def test_fingerprint_previous_reuse_matches_fresh_and_skips_full_blocks(monkeypatch):
    monkeypatch.setattr(hash_index, "FINGERPRINT_BLOCK_ROWS", 10)
    data = np.random.default_rng(2).normal(size=(47, 3))
    previous = hash_index.dataset_fingerprint(recording_store(data, []), 25)

    reads = []
    incremental = hash_index.dataset_fingerprint(
        recording_store(data, reads), 47, previous=previous
    )
    fresh = hash_index.dataset_fingerprint(recording_store(data, []), 47)
    assert incremental == fresh
    # the two complete blocks (rows 0-19) are reused; reading starts at row 20
    assert reads
    assert min(start for start, _ in reads) == 20
    assert max(stop for _, stop in reads) == 47


def test_fingerprint_ignores_previous_with_other_block_size(monkeypatch):
    monkeypatch.setattr(hash_index, "FINGERPRINT_BLOCK_ROWS", 10)
    data = np.random.default_rng(3).normal(size=(30, 2))
    previous = hash_index.dataset_fingerprint(recording_store(data, []), 20)
    monkeypatch.setattr(hash_index, "FINGERPRINT_BLOCK_ROWS", 8)
    reads = []
    result = hash_index.dataset_fingerprint(
        recording_store(data, reads), 30, previous=previous
    )
    assert result == hash_index.dataset_fingerprint(recording_store(data, []), 30)
    assert min(start for start, _ in reads) == 0


# --- delta segments (format 3) ------------------------------------------------


def _segment_sizes(index_dir, perturbation=0):
    metadata = json.loads((index_dir / METADATA_FILE).read_text())
    base = np.load(index_dir / metadata["hash_files"][perturbation], mmap_mode="r")
    deltas = [
        np.load(index_dir / name, mmap_mode="r")
        for name in metadata["delta_files"][perturbation]
    ]
    return base.shape[0], [d.shape[0] for d in deltas]


def test_saves_write_delta_segments_and_compact_geometrically(tmp_path, monkeypatch):
    """Many small saves: each writes a delta instead of rewriting the base table,
    deltas stay few and geometrically sized, and the answers match a fresh build."""
    monkeypatch.setattr(MultiHashing, "_pending_limit", lambda self: 30)
    path = hdf5_dataset(tmp_path, DATA[:150])
    index_dir = tmp_path / "idx"
    dda = MultiHashing(dataset=path, index_dir=index_dir, **MH_KWARGS)
    dda.get_dataset_unique_structures()
    base_file = json.loads((index_dir / METADATA_FILE).read_text())["hash_files"][0]

    saves = []
    original_save = dda._save_index
    monkeypatch.setattr(dda, "_save_index", lambda: (saves.append(1), original_save()))
    for vector in DATA[150:]:
        dda.input_vector = vector
        assert dda.duplicate_check() == _reference_duplicate(dda, vector)
        dda.add_input_vector_to_dda()
        metadata = json.loads((index_dir / METADATA_FILE).read_text())
        assert metadata["format_version"] == 3
        for perturbation in range(dda.perturbations):
            n_deltas = len(metadata["delta_files"][perturbation])
            assert n_deltas <= 8  # O(log n) segments
    dda.flush()
    assert len(saves) > 10

    # the hashes of a perturbation are spread over distinct, sorted segments
    for perturbation in range(dda.perturbations):
        segments = dda._segments(perturbation)
        joined = np.concatenate([np.asarray(s) for s in segments])
        assert len(np.unique(joined)) == len(joined)
        assert all(
            np.all(np.diff(np.asarray(s).astype(np.int64)) != 0) for s in segments
        )
        in_memory = MultiHashing(dataset_array=DATA, **MH_KWARGS)
        in_memory.get_dataset_unique_structures()
        assert np.array_equal(np.sort(joined), in_memory.sorted_hashes[perturbation])

    # deltas grow geometrically: each is more than COMPACTION_RATIO x the next newer one
    from deduplicate_lib.plugins.duplicate_detection_algorithms.multi_hashing import (
        COMPACTION_RATIO,
    )

    for perturbation in range(dda.perturbations):
        _, sizes = _segment_sizes(index_dir, perturbation)
        assert all(a > COMPACTION_RATIO * b for a, b in itertools.pairwise(sizes))

    dda.get_dataset_unique_structures()
    assert np.array_equal(dda.get_unique_vector_mask(), fresh_mask())
    no_hashing(monkeypatch)
    reopened = MultiHashing(dataset=path, index_dir=index_dir, **MH_KWARGS)
    reopened.get_dataset_unique_structures()
    assert np.array_equal(reopened.get_unique_vector_mask(), fresh_mask())
    for vector in DATA[:20]:  # lookups against the reopened segments
        reopened.input_vector = vector
        assert reopened.duplicate_check()
    files = {f.name for f in index_dir.iterdir()}
    metadata = json.loads((index_dir / METADATA_FILE).read_text())
    referenced = {metadata["counts_file"], *metadata["hash_files"]}
    referenced |= {name for names in metadata["delta_files"] for name in names}
    referenced |= {fence_name(name) for name in referenced if name.startswith("p")}
    assert files == {METADATA_FILE, *referenced}  # superseded segments are deleted
    assert (
        base_file not in files
    )  # the deltas grew to half the base and were merged into it


def _reference_duplicate(dda, vector):
    """duplicate_check() computed from all hashes so far, without segments."""
    hashes = dda.create_hash_vector(vector)
    seen = 0
    for perturbation, value in enumerate(hashes):
        everything = np.concatenate(
            [np.asarray(s) for s in dda._segments(perturbation)]
            + [dda._pending_hashes[perturbation].sorted()]
        )
        seen += bool(np.isin(value, everything))
    return seen / dda.perturbations >= dda.acceptance_threshold


def test_a_save_writes_only_the_new_hashes(tmp_path, monkeypatch):
    index_dir = tmp_path / "idx"
    path = hdf5_dataset(tmp_path, DATA[:250])
    dda = MultiHashing(dataset=path, index_dir=index_dir, **MH_KWARGS)
    dda.get_dataset_unique_structures()
    base_before, _ = _segment_sizes(index_dir)
    written = []
    original = HashIndex._save
    monkeypatch.setattr(
        HashIndex,
        "_save",
        lambda self, name, array: (
            written.append(len(array)),
            original(self, name, array),
        ),
    )
    dda.add_vectors(DATA[250:260])
    dda.flush()
    tables = list(written[:-1])  # the last write is the counts file
    assert max(tables) < base_before / 2  # small deltas, not the whole table
    base_after, deltas = _segment_sizes(index_dir)
    assert base_after == base_before and len(deltas) == 1


def test_format_2_index_opens_and_upgrades(tmp_path, monkeypatch):
    path = hdf5_dataset(tmp_path, DATA[:200])
    index_dir = tmp_path / "idx"
    MultiHashing(
        dataset=path, index_dir=index_dir, **MH_KWARGS
    ).get_dataset_unique_structures()
    # rewrite the metadata as format 2 wrote it: no delta_files
    metadata = json.loads((index_dir / METADATA_FILE).read_text())
    metadata["format_version"] = 2
    del metadata["delta_files"]
    (index_dir / METADATA_FILE).write_text(json.dumps(metadata))

    reopened = MultiHashing(dataset=path, index_dir=index_dir, **MH_KWARGS)
    with monkeypatch.context() as patch:
        no_hashing(patch)
        reopened.get_dataset_unique_structures()
    reopened.add_vectors(DATA[200:])
    reopened.flush()
    reopened.get_dataset_unique_structures()
    assert np.array_equal(reopened.get_unique_vector_mask(), fresh_mask())
    upgraded = json.loads((index_dir / METADATA_FILE).read_text())
    assert upgraded["format_version"] == 3 and "delta_files" in upgraded


def test_crash_before_commit_keeps_previous_segments(tmp_path, monkeypatch):
    monkeypatch.setattr(MultiHashing, "_pending_limit", lambda self: 10**9)
    path = hdf5_dataset(tmp_path, DATA[:200])
    index_dir = tmp_path / "idx"
    dda = MultiHashing(dataset=path, index_dir=index_dir, **MH_KWARGS)
    dda.get_dataset_unique_structures()
    dda.add_vectors(DATA[200:250])
    dda.flush()  # one delta per perturbation
    committed = json.loads((index_dir / METADATA_FILE).read_text())

    dda.add_vectors(DATA[250:])

    def crash(*_args, **_kwargs):
        raise RuntimeError("crash")

    monkeypatch.setattr(HashIndex, "commit", crash)
    with pytest.raises(RuntimeError):
        dda.flush()  # segments staged, never committed
    assert json.loads((index_dir / METADATA_FILE).read_text()) == committed
    monkeypatch.undo()

    reopened = MultiHashing(dataset=path, index_dir=index_dir, **MH_KWARGS)
    reopened.get_dataset_unique_structures()  # recovers the unflushed rows from the HDF5 file
    assert np.array_equal(reopened.get_unique_vector_mask(), fresh_mask())
    files = {f.name for f in index_dir.iterdir()}
    metadata = json.loads((index_dir / METADATA_FILE).read_text())
    referenced = {metadata["counts_file"], *metadata["hash_files"]}
    referenced |= {name for names in metadata["delta_files"] for name in names}
    referenced |= {fence_name(name) for name in referenced if name.startswith("p")}
    assert files == {METADATA_FILE, *referenced}  # the staged files were cleaned up


def test_segments_have_fence_sidecars(tmp_path, monkeypatch):
    """Every committed segment has a fence file that loads with it; a missing
    fence is recomputed, and lookups agree either way."""
    monkeypatch.setattr(MultiHashing, "_pending_limit", lambda self: 10**9)
    path = hdf5_dataset(tmp_path, DATA[:200])
    index_dir = tmp_path / "idx"
    dda = MultiHashing(dataset=path, index_dir=index_dir, **MH_KWARGS)
    dda.get_dataset_unique_structures()
    dda.add_vectors(DATA[200:])
    dda.flush()
    metadata = json.loads((index_dir / METADATA_FILE).read_text())
    segment_names = [*metadata["hash_files"]]
    segment_names += [name for names in metadata["delta_files"] for name in names]
    assert any(metadata["delta_files"])
    for name in segment_names:
        assert (index_dir / fence_name(name)).exists()

    index = HashIndex(index_dir, metadata)
    stored = index.load_table(0)
    # the stored fence (computed before writing) matches the file's page layout
    assert np.array_equal(stored.fence, SortedSegment(stored.table).fence)
    (index_dir / fence_name(metadata["hash_files"][0])).unlink()
    recomputed = index.load_table(0)
    assert np.array_equal(recomputed.fence, stored.fence)

    reopened = MultiHashing(dataset=path, index_dir=index_dir, **MH_KWARGS)
    reopened.get_dataset_unique_structures()
    assert np.array_equal(reopened.get_unique_vector_mask(), fresh_mask())
    for vector in DATA[::37]:
        reopened.input_vector = vector
        assert reopened.duplicate_check() == _reference_duplicate(reopened, vector)


def test_saves_append_counts_instead_of_rewriting_them(tmp_path, monkeypatch):
    monkeypatch.setattr(MultiHashing, "_pending_limit", lambda self: 10**9)
    path = hdf5_dataset(tmp_path, DATA[:200])
    index_dir = tmp_path / "idx"
    dda = MultiHashing(dataset=path, index_dir=index_dir, **MH_KWARGS)
    dda.get_dataset_unique_structures()
    counts_file = json.loads((index_dir / METADATA_FILE).read_text())["counts_file"]
    assert counts_file.endswith(".bin")

    for start in range(200, len(DATA), 25):
        dda.add_vectors(DATA[start : start + 25])
        dda.flush()
        metadata = json.loads((index_dir / METADATA_FILE).read_text())
        assert metadata["counts_file"] == counts_file  # appended, not rewritten
        assert (index_dir / counts_file).stat().st_size == 4 * metadata["n_vectors"]
    dda.get_dataset_unique_structures()
    assert np.array_equal(dda.get_unique_vector_mask(), fresh_mask())
    reopened = MultiHashing(dataset=path, index_dir=index_dir, **MH_KWARGS)
    with monkeypatch.context() as patch:
        no_hashing(patch)
        reopened.get_dataset_unique_structures()
    assert np.array_equal(reopened.get_unique_vector_mask(), fresh_mask())


def test_interrupted_counts_append_is_cut_off(tmp_path, monkeypatch):
    monkeypatch.setattr(MultiHashing, "_pending_limit", lambda self: 10**9)
    path = hdf5_dataset(tmp_path, DATA[:200])
    index_dir = tmp_path / "idx"
    dda = MultiHashing(dataset=path, index_dir=index_dir, **MH_KWARGS)
    dda.get_dataset_unique_structures()
    committed = json.loads((index_dir / METADATA_FILE).read_text())

    dda.add_vectors(DATA[200:250])

    def crash(*_args, **_kwargs):
        raise RuntimeError("crash")

    monkeypatch.setattr(HashIndex, "_write_metadata", crash)
    with pytest.raises(RuntimeError):
        dda.flush()  # counts appended, never committed
    counts_path = index_dir / committed["counts_file"]
    assert counts_path.stat().st_size == 4 * 250
    assert json.loads((index_dir / METADATA_FILE).read_text()) == committed
    monkeypatch.undo()
    monkeypatch.setattr(MultiHashing, "_pending_limit", lambda self: 10**9)

    reopened = MultiHashing(dataset=path, index_dir=index_dir, **MH_KWARGS)
    reopened.get_dataset_unique_structures()  # re-adds rows 200-250 from the HDF5 file
    reopened.add_vectors(DATA[250:])
    reopened.flush()
    metadata = json.loads((index_dir / METADATA_FILE).read_text())
    assert counts_path.stat().st_size == 4 * metadata["n_vectors"] == 4 * len(DATA)
    reopened.get_dataset_unique_structures()
    assert np.array_equal(reopened.get_unique_vector_mask(), fresh_mask())


def test_npy_counts_still_open_and_become_appendable(tmp_path):
    path = hdf5_dataset(tmp_path, DATA[:200])
    index_dir = tmp_path / "idx"
    MultiHashing(
        dataset=path, index_dir=index_dir, **MH_KWARGS
    ).get_dataset_unique_structures()
    metadata = json.loads((index_dir / METADATA_FILE).read_text())
    counts = np.fromfile(index_dir / metadata["counts_file"], dtype="<u4")
    np.save(index_dir / "counts.g999.npy", counts)  # as earlier versions wrote them
    metadata["counts_file"] = "counts.g999.npy"
    (index_dir / METADATA_FILE).write_text(json.dumps(metadata))

    reopened = MultiHashing(dataset=path, index_dir=index_dir, **MH_KWARGS)
    reopened.add_vectors(DATA[200:])
    reopened.flush()
    upgraded = json.loads((index_dir / METADATA_FILE).read_text())
    assert upgraded["counts_file"].endswith(".bin")
    reopened.get_dataset_unique_structures()
    assert np.array_equal(reopened.get_unique_vector_mask(), fresh_mask())
