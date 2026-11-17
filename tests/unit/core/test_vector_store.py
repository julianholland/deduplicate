import h5py
import numpy as np
import pytest

from deduplicate_lib.core.vector_store import (
    CallableStore,
    HDF5Store,
    InMemoryStore,
    NpyStore,
    VectorStore,
    open_store,
)

DATA = np.arange(23 * 4, dtype=np.float64).reshape(23, 4) / 3


def make_store(kind, tmp_path, data=DATA):
    if kind == "memory":
        return InMemoryStore(data)
    if kind == "npy":
        np.save(tmp_path / "data.npy", data)
        return NpyStore(tmp_path / "data.npy")
    if kind == "hdf5":
        with h5py.File(tmp_path / "data.h5", "w") as h5:
            h5.create_dataset("vectors", data=data, maxshape=(None, data.shape[1]))
        return HDF5Store(tmp_path / "data.h5")
    if kind == "callable":
        return CallableStore(len(data), data.shape[1], lambda a, b: data[a:b])
    raise ValueError(kind)


KINDS = ["memory", "npy", "hdf5", "callable"]


@pytest.mark.parametrize("kind", KINDS)
def test_shape_dtype_and_read(kind, tmp_path):
    store = make_store(kind, tmp_path)
    assert isinstance(store, VectorStore)
    assert store.shape == DATA.shape
    assert len(store) == 23
    assert store.dim == 4
    assert store.dtype == np.float64
    assert store.nbytes == DATA.nbytes
    assert np.array_equal(store.read(5, 9), DATA[5:9])
    assert store.in_memory == (kind == "memory")
    assert type(store).__name__ in repr(store)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("chunk_rows", [1, 5, 23, 100])
def test_iter_chunks_covers_store_in_order(kind, chunk_rows, tmp_path):
    store = make_store(kind, tmp_path)
    chunks = list(store.iter_chunks(chunk_rows))
    assert [start for start, _ in chunks] == list(range(0, 23, chunk_rows))
    assert np.array_equal(np.vstack([rows for _, rows in chunks]), DATA)


@pytest.mark.parametrize("kind", KINDS)
def test_take_any_order(kind, tmp_path):
    store = make_store(kind, tmp_path)
    indices = np.array([22, 0, 7, 7, 13, 1])
    assert np.array_equal(store.take(indices, chunk_rows=4), DATA[indices])
    assert store.take(np.array([], dtype=int)).shape == (0, 4)
    with pytest.raises(IndexError):
        store.take(np.array([23]))


@pytest.mark.parametrize("kind", KINDS)
def test_sample_is_distinct_ordered_and_seeded(kind, tmp_path):
    store = make_store(kind, tmp_path)
    sample = store.sample(10, seed=1)
    assert sample.shape == (10, 4)
    rows = [int(np.where((row == DATA).all(axis=1))[0][0]) for row in sample]
    assert rows == sorted(set(rows))
    assert np.array_equal(sample, store.sample(10, seed=1))
    assert store.sample(100).shape == (23, 4)  # capped at the store size


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("suffix", [".npy", ".h5", ".hdf5"])
def test_write_subset_round_trips(kind, suffix, tmp_path):
    store = make_store(kind, tmp_path)
    mask = np.zeros(23, dtype=bool)
    mask[[0, 3, 4, 20, 22]] = True
    written = store.write_subset(mask, tmp_path / f"out{suffix}", chunk_rows=3)
    assert written.path == tmp_path / f"out{suffix}"
    assert np.array_equal(written.read(0, len(written)), DATA[mask])


def test_write_subset_empty_mask_and_errors(tmp_path):
    store = InMemoryStore(DATA)
    for suffix in (".npy", ".h5"):
        written = store.write_subset(np.zeros(23, bool), tmp_path / f"empty{suffix}")
        assert written.shape == (0, 4)
    with pytest.raises(ValueError, match="one entry per vector"):
        store.write_subset(np.ones(3, bool), tmp_path / "x.npy")
    with pytest.raises(ValueError, match="Unsupported output format"):
        store.write_subset(np.ones(23, bool), tmp_path / "x.csv")


@pytest.mark.parametrize("kind", ["npy", "hdf5"])
def test_default_output_path_mirrors_input(kind, tmp_path):
    store = make_store(kind, tmp_path)
    assert store.default_output_path() == store.path.with_name(
        f"data_unique{store.path.suffix}"
    )


@pytest.mark.parametrize("kind", ["memory", "callable"])
def test_default_output_path_needs_a_file(kind, tmp_path):
    with pytest.raises(ValueError, match="pass an explicit output path"):
        make_store(kind, tmp_path).default_output_path()


@pytest.mark.parametrize("kind", ["memory", "npy", "callable"])
def test_read_only_stores_refuse_append(kind, tmp_path):
    with pytest.raises(NotImplementedError, match="read-only"):
        make_store(kind, tmp_path).append(DATA[:1])


def test_hdf5_append_and_create(tmp_path):
    store = HDF5Store.create(tmp_path / "new.h5", dim=4)
    assert store.shape == (0, 4)
    store.append(DATA[:5])
    store.append(DATA[5])
    assert store.shape == (6, 4)
    assert np.array_equal(HDF5Store(tmp_path / "new.h5").read(0, 6), DATA[:6])
    with pytest.raises(ValueError, match="wrong length"):
        store.append(np.zeros((1, 3)))


def test_hdf5_append_needs_resizable_dataset(tmp_path):
    with h5py.File(tmp_path / "fixed.h5", "w") as h5:
        h5.create_dataset("vectors", data=DATA)
    with pytest.raises(NotImplementedError, match="not resizable"):
        HDF5Store(tmp_path / "fixed.h5").append(DATA[:1])


def test_hdf5_key_errors(tmp_path):
    with h5py.File(tmp_path / "keys.h5", "w") as h5:
        h5.create_dataset("descriptors", data=DATA)
        h5.create_dataset("flat", data=np.arange(3))
    assert HDF5Store(tmp_path / "keys.h5", key="descriptors").shape == (23, 4)
    with pytest.raises(KeyError, match="available"):
        HDF5Store(tmp_path / "keys.h5")
    with pytest.raises(ValueError, match="not a 2-D"):
        HDF5Store(tmp_path / "keys.h5", key="flat")


def test_npy_and_memory_need_2d(tmp_path):
    np.save(tmp_path / "flat.npy", np.arange(3))
    with pytest.raises(ValueError, match="2-D"):
        NpyStore(tmp_path / "flat.npy")
    with pytest.raises(ValueError, match="2-D"):
        InMemoryStore(np.arange(3))


def test_callable_store_checks_fetch_shape():
    store = CallableStore(5, 4, lambda a, b: np.zeros((b - a + 1, 4)))
    with pytest.raises(ValueError, match="returned shape"):
        store.read(0, 2)


def test_callable_store_casts_dtype():
    store = CallableStore(2, 2, lambda a, b: [[1, 2], [3, 4]][a:b], dtype=np.float32)
    assert store.read(0, 2).dtype == np.float32


def test_open_store_dispatch(tmp_path):
    assert isinstance(open_store(DATA), InMemoryStore)
    np.save(tmp_path / "a.npy", DATA)
    assert isinstance(open_store(tmp_path / "a.npy"), NpyStore)
    assert isinstance(open_store(str(tmp_path / "a.npy")), NpyStore)
    with h5py.File(tmp_path / "a.hdf5", "w") as h5:
        h5.create_dataset("x", data=DATA)
    assert isinstance(open_store(tmp_path / "a.hdf5", key="x"), HDF5Store)
    store = InMemoryStore(DATA)
    assert open_store(store) is store

    with pytest.raises(FileNotFoundError):
        open_store(tmp_path / "missing.npy")
    (tmp_path / "a.csv").write_text("1,2")
    with pytest.raises(ValueError, match="CallableStore"):
        open_store(tmp_path / "a.csv")
    with pytest.raises(TypeError):
        open_store([[1.0, 2.0]])


def test_missing_h5py_gives_install_hint(monkeypatch, tmp_path):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "h5py":
            raise ImportError("no h5py")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"deduplicate_lib\[hdf5\]"):
        HDF5Store(tmp_path / "x.h5")


# --- sample_blocks ---

BLOCK_DATA = np.arange(95 * 3, dtype=np.float64).reshape(95, 3)


def _row_ids(sample):
    return (sample[:, 0] / 3).astype(int)


@pytest.mark.parametrize("kind", ["memory", "npy"])
def test_sample_blocks_contiguous_ordered_blocks(kind, tmp_path):
    store = make_store(kind, tmp_path, BLOCK_DATA)
    sample = store.sample_blocks(30, block_rows=10, seed=3)
    assert sample.shape == (30, 3)
    ids = _row_ids(sample)
    assert np.array_equal(sample, BLOCK_DATA[ids])
    for block in ids.reshape(3, 10):
        assert block[0] % 10 == 0
        assert np.array_equal(block, np.arange(block[0], block[0] + 10))
    assert (np.diff(ids) > 0).all()  # dataset order


@pytest.mark.parametrize("kind", ["memory", "npy"])
def test_sample_blocks_truncates_to_k(kind, tmp_path):
    store = make_store(kind, tmp_path, BLOCK_DATA)
    sample = store.sample_blocks(25, block_rows=10, seed=3)
    assert len(sample) == 25
    assert (np.diff(_row_ids(sample)) > 0).all()


@pytest.mark.parametrize("k", [95, 96, 1000])
def test_sample_blocks_whole_dataset_when_n_le_k(k):
    sample = InMemoryStore(BLOCK_DATA).sample_blocks(k, block_rows=10)
    assert np.array_equal(sample, BLOCK_DATA)


def test_sample_blocks_seed_determinism():
    store = InMemoryStore(BLOCK_DATA)
    a = store.sample_blocks(30, block_rows=10, seed=1)
    assert np.array_equal(a, store.sample_blocks(30, block_rows=10, seed=1))
    seeds_differ = [
        not np.array_equal(a, store.sample_blocks(30, block_rows=10, seed=s))
        for s in range(2, 8)
    ]
    assert any(seeds_differ)


def test_sample_blocks_matches_between_store_types(tmp_path):
    mem = InMemoryStore(BLOCK_DATA).sample_blocks(40, block_rows=10, seed=9)
    npy = make_store("npy", tmp_path, BLOCK_DATA).sample_blocks(
        40, block_rows=10, seed=9
    )
    assert np.array_equal(mem, npy)


@pytest.mark.parametrize("block_rows", [0, -1])
def test_sample_blocks_rejects_bad_block_rows(block_rows):
    with pytest.raises(ValueError, match="block_rows"):
        InMemoryStore(BLOCK_DATA).sample_blocks(10, block_rows=block_rows)


# --- write_subset never overwrites its own input ---


@pytest.mark.parametrize("kind", ["npy", "hdf5"])
def test_write_subset_refuses_to_overwrite_input(kind, tmp_path):
    store = make_store(kind, tmp_path)
    before = store.path.read_bytes()
    with pytest.raises(ValueError, match="Refusing"):
        store.write_subset(np.ones(len(store), dtype=bool), store.path)
    assert store.path.read_bytes() == before
    assert np.array_equal(store.read(0, len(store)), DATA)


# --- take() ---

TAKE_INDICES = [20, 3, 3, 0, 22, 7, 20]


@pytest.mark.parametrize("kind", ["memory", "npy", "hdf5", "callable"])
def test_take_unsorted_with_duplicates(kind, tmp_path):
    store = make_store(kind, tmp_path)
    assert np.array_equal(store.take(TAKE_INDICES), DATA[TAKE_INDICES])
    # small chunks exercise the chunked base implementation
    assert np.array_equal(store.take(TAKE_INDICES, chunk_rows=4), DATA[TAKE_INDICES])
    assert store.take([]).shape == (0, DATA.shape[1])


@pytest.mark.parametrize("kind", ["npy", "hdf5", "callable"])
@pytest.mark.parametrize("bad", [[23], [0, -1], [100]])
def test_take_out_of_range_raises(kind, bad, tmp_path):
    store = make_store(kind, tmp_path)
    with pytest.raises(IndexError):
        store.take(bad)


def test_take_sparse_reads_only_the_spanned_rows():
    n = 200_000
    reads = []

    def fetch(start, stop):
        reads.append(stop - start)
        return np.arange(start, stop, dtype=np.float64).reshape(-1, 1) * np.ones((1, 2))

    store = CallableStore(n, 2, fetch)
    indices = [150_000, 5, 70_000]
    rows = store.take(indices)
    assert np.array_equal(rows[:, 0], indices)
    # one tiny read per chunk holding a requested row, not 65536 rows each
    assert sum(reads) == 3
    assert len(reads) == 3


def test_take_reads_span_within_a_chunk():
    reads = []

    def fetch(start, stop):
        reads.append((start, stop))
        return np.zeros((stop - start, 1))

    CallableStore(1000, 1, fetch).take([10, 40, 11], chunk_rows=100)
    assert reads == [(10, 41)]
