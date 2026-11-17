"""Where a dataset's vectors live: in memory, in ``.npy``/HDF5 files, or behind
a callable (e.g. a database adapter).

Algorithms read stores in chunks of rows, so a dataset never has to fit in
memory as a whole.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

NPY_SUFFIXES = (".npy",)
HDF5_SUFFIXES = (".h5", ".hdf5")
DEFAULT_HDF5_KEY = "vectors"


def _import_h5py():
    try:
        import h5py
    except ImportError as error:
        raise ImportError(
            "HDF5 datasets need h5py; install it with `pip install deduplicate_lib[hdf5]`."
        ) from error
    return h5py


class VectorStore(ABC):
    """A 2-D dataset of vectors (rows), read in chunks.

    Subclasses implement ``shape``, ``dtype`` and ``read``; everything else is
    built on those.  ``path`` is the backing file, if any.
    """

    path: Path | None = None
    in_memory: bool = False

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """``(number of vectors, vector length)``."""

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """dtype of the stored values."""

    @abstractmethod
    def read(self, start: int, stop: int) -> np.ndarray:
        """Rows ``start`` to ``stop`` (exclusive) as an in-memory array."""

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def dim(self) -> int:
        return self.shape[1]

    @property
    def nbytes(self) -> int:
        return self.shape[0] * self.shape[1] * self.dtype.itemsize

    def __repr__(self) -> str:
        where = f", path={str(self.path)!r}" if self.path is not None else ""
        return f"{type(self).__name__}(shape={self.shape}{where})"

    def iter_chunks(
        self, chunk_rows: int, start: int = 0, stop: int | None = None
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(row, rows)`` covering rows ``start`` to ``stop`` (default: all) in order."""
        stop = len(self) if stop is None else min(stop, len(self))
        for row in range(start, stop, chunk_rows):
            yield row, self.read(row, min(row + chunk_rows, stop))

    def _check_indices(self, indices) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size and (indices.min() < 0 or indices.max() >= len(self)):
            raise IndexError("take() indices out of range for the store.")
        return indices

    def take(self, indices: np.ndarray, chunk_rows: int = 65536) -> np.ndarray:
        """Rows at ``indices`` (in the given order).

        Within each chunk of ``chunk_rows`` rows, only the span from the first
        to the last requested row is read, so a sparse sample of a large store
        reads little more than the rows it returns.
        """
        indices = self._check_indices(indices)
        out = np.empty((indices.size, self.dim), dtype=self.dtype)
        order = np.argsort(indices, kind="stable")
        sorted_indices = indices[order]
        bounds = np.searchsorted(
            sorted_indices, np.arange(0, len(self) + chunk_rows, chunk_rows)
        )
        for chunk_number in range(len(bounds) - 1):
            lo, hi = bounds[chunk_number], bounds[chunk_number + 1]
            if lo == hi:
                continue
            first, last = sorted_indices[lo], sorted_indices[hi - 1]
            rows = self.read(int(first), int(last) + 1)
            out[order[lo:hi]] = rows[sorted_indices[lo:hi] - first]
        return out

    def sample(self, k: int, seed: int = 803) -> np.ndarray:
        """``k`` distinct random rows, kept in dataset order."""
        k = min(k, len(self))
        rng = np.random.default_rng(seed)
        return self.take(np.sort(rng.choice(len(self), size=k, replace=False)))

    def sample_blocks(
        self, k: int, block_rows: int = 100, seed: int = 803
    ) -> np.ndarray:
        """``k`` rows taken as contiguous blocks of ``block_rows`` from random places.

        The dataset is split into aligned slots of ``block_rows`` rows; random
        slots are read whole (in dataset order) until ``k`` rows are collected.
        Unlike :meth:`sample`, this keeps near-duplicates that sit next to each
        other (e.g. consecutive frames of a trajectory) together in the sample.
        Returns every row when the dataset has at most ``k``.
        """
        if block_rows < 1:
            raise ValueError("block_rows must be a positive integer.")
        n = len(self)
        if k >= n:
            return self.read(0, n)
        n_slots = -(-n // block_rows)
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_slots)
        # slots are full except possibly the last one, so take slots in random
        # order until they hold at least k rows
        slot_sizes = np.minimum(block_rows, n - order * block_rows)
        n_blocks = int(np.searchsorted(np.cumsum(slot_sizes), k)) + 1
        slots = np.sort(order[:n_blocks])
        rows = np.concatenate(
            [
                self.read(slot * block_rows, min((slot + 1) * block_rows, n))
                for slot in slots
            ]
        )
        return rows[:k]

    def append(self, vectors: np.ndarray) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} is read-only; store the dataset in HDF5 "
            "(HDF5Store) to append vectors to it."
        )

    def default_output_path(self) -> Path:
        """Where ``write_subset`` writes when no path is given: ``<input>_unique.<ext>``."""
        if self.path is None:
            raise ValueError(
                f"{type(self).__name__} has no file to name the output after; "
                "pass an explicit output path (deduplicate(out=...))."
            )
        return self.path.with_name(f"{self.path.stem}_unique{self.path.suffix}")

    def write_subset(
        self, mask: np.ndarray, path: str | Path, chunk_rows: int = 65536
    ) -> "VectorStore":
        """Stream the rows where ``mask`` is True to a new file and open it.

        The format follows the output path's extension (``.npy``, ``.h5``/``.hdf5``).
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(self),):
            raise ValueError("mask must have one entry per vector in the store.")
        path = Path(path)
        if self.path is not None and path.exists() and path.samefile(self.path):
            raise ValueError(
                f"Refusing to write the unique vectors over the input dataset {path}; "
                "choose a different output path."
            )
        n_out = int(mask.sum())

        if path.suffix in NPY_SUFFIXES:
            out = np.lib.format.open_memmap(
                path, mode="w+", dtype=self.dtype, shape=(n_out, self.dim)
            )
            self._copy_masked(mask, out, chunk_rows)
            out.flush()
            del out
            return NpyStore(path)

        if path.suffix in HDF5_SUFFIXES:
            h5py = _import_h5py()
            with h5py.File(path, "w") as h5:
                out = h5.create_dataset(
                    DEFAULT_HDF5_KEY,
                    shape=(n_out, self.dim),
                    maxshape=(None, self.dim),
                    dtype=self.dtype,
                    chunks=(max(1, min(n_out, 4096)), self.dim) if n_out else True,
                )
                self._copy_masked(mask, out, chunk_rows)
            return HDF5Store(path, key=DEFAULT_HDF5_KEY)

        raise ValueError(
            f"Unsupported output format {path.suffix!r}; use one of "
            f"{NPY_SUFFIXES + HDF5_SUFFIXES}."
        )

    def _copy_masked(self, mask: np.ndarray, out, chunk_rows: int) -> None:
        written = 0
        for start, rows in self.iter_chunks(chunk_rows):
            selected = rows[mask[start : start + len(rows)]]
            if len(selected):
                out[written : written + len(selected)] = selected
                written += len(selected)


class InMemoryStore(VectorStore):
    """A store over an in-memory array (no copy is made)."""

    in_memory = True

    def __init__(self, array: np.ndarray) -> None:
        array = np.asarray(array)
        if array.ndim != 2:
            raise ValueError("InMemoryStore needs a 2-D array of vectors.")
        self.array = array

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    def read(self, start: int, stop: int) -> np.ndarray:
        return self.array[start:stop]

    def take(self, indices: np.ndarray, chunk_rows: int = 65536) -> np.ndarray:  # noqa: ARG002
        return self.array[np.asarray(indices, dtype=np.int64)]


class NpyStore(VectorStore):
    """A ``.npy`` file, memory-mapped read-only."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._array = np.load(self.path, mmap_mode="r")
        if self._array.ndim != 2:
            raise ValueError(f"{self.path} does not hold a 2-D array of vectors.")

    @property
    def shape(self) -> tuple[int, int]:
        return self._array.shape

    @property
    def dtype(self) -> np.dtype:
        return self._array.dtype

    def read(self, start: int, stop: int) -> np.ndarray:
        return np.array(self._array[start:stop])

    def take(self, indices: np.ndarray, chunk_rows: int = 65536) -> np.ndarray:  # noqa: ARG002
        # fancy indexing a memmap touches only the pages holding those rows
        return np.array(self._array[self._check_indices(indices)])


class HDF5Store(VectorStore):
    """A 2-D dataset (``key``) in an HDF5 file.  Appendable if the dataset is resizable."""

    def __init__(self, path: str | Path, key: str = DEFAULT_HDF5_KEY) -> None:
        self._h5py = _import_h5py()
        self.path = Path(path)
        self.key = key
        with self._h5py.File(self.path, "r") as h5:
            if key not in h5:
                raise KeyError(
                    f"{self.path} has no dataset {key!r}; available: {list(h5.keys())}. "
                    "Pass the dataset name as key=..."
                )
            dataset = h5[key]
            if dataset.ndim != 2:
                raise ValueError(f"{self.path}:{key} is not a 2-D array of vectors.")
            self._shape = tuple(dataset.shape)
            self._dtype = dataset.dtype

    @classmethod
    def create(
        cls,
        path: str | Path,
        dim: int,
        dtype=np.float64,
        key: str = DEFAULT_HDF5_KEY,
    ) -> "HDF5Store":
        """Create an empty, appendable HDF5 dataset of vectors of length ``dim``."""
        h5py = _import_h5py()
        with h5py.File(path, "w") as h5:
            h5.create_dataset(
                key,
                shape=(0, dim),
                maxshape=(None, dim),
                dtype=dtype,
                chunks=(4096, dim),
            )
        return cls(path, key=key)

    def __repr__(self) -> str:
        return (
            f"HDF5Store(shape={self.shape}, path={str(self.path)!r}, key={self.key!r})"
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def read(self, start: int, stop: int) -> np.ndarray:
        with self._h5py.File(self.path, "r") as h5:
            return h5[self.key][start:stop]

    def take(self, indices: np.ndarray, chunk_rows: int = 65536) -> np.ndarray:  # noqa: ARG002
        # one open and a point selection (h5py needs increasing, unique indices)
        indices = self._check_indices(indices)
        unique, inverse = np.unique(indices, return_inverse=True)
        with self._h5py.File(self.path, "r") as h5:
            rows = h5[self.key][unique] if unique.size else np.empty((0, self.dim))
        return np.asarray(rows, dtype=self.dtype)[inverse]

    def append(self, vectors: np.ndarray) -> None:
        vectors = np.atleast_2d(np.asarray(vectors))
        if vectors.shape[1] != self.dim:
            raise ValueError("Appended vectors have the wrong length for this store.")
        with self._h5py.File(self.path, "a") as h5:
            dataset = h5[self.key]
            if dataset.maxshape[0] is not None:
                raise NotImplementedError(
                    f"{self.path}:{self.key} is not resizable; create it with "
                    "maxshape=(None, dim) (e.g. HDF5Store.create) to append to it."
                )
            n = dataset.shape[0]
            dataset.resize(n + len(vectors), axis=0)
            dataset[n:] = vectors
            self._shape = tuple(dataset.shape)


class CallableStore(VectorStore):
    """Vectors served by ``fetch(start, stop) -> array of rows``.

    The adapter for anything else, such as a structure database:
    ``CallableStore(n, dim, lambda start, stop: load_descriptors(start, stop))``.
    Read-only; ``deduplicate()`` needs an explicit ``out=`` path.
    """

    def __init__(
        self,
        n_vectors: int,
        dim: int,
        fetch: Callable[[int, int], np.ndarray],
        dtype=np.float64,
    ) -> None:
        self._shape = (int(n_vectors), int(dim))
        self._dtype = np.dtype(dtype)
        self.fetch = fetch

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def read(self, start: int, stop: int) -> np.ndarray:
        rows = np.asarray(self.fetch(start, stop), dtype=self._dtype)
        if rows.shape != (stop - start, self.dim):
            raise ValueError(
                f"fetch({start}, {stop}) returned shape {rows.shape}; "
                f"expected {(stop - start, self.dim)}."
            )
        return rows


def open_store(dataset, key: str = DEFAULT_HDF5_KEY) -> VectorStore:
    """Open ``dataset`` as a store: an array, a ``.npy``/``.h5``/``.hdf5`` path, or a store."""
    if isinstance(dataset, VectorStore):
        return dataset
    if isinstance(dataset, np.ndarray):
        return InMemoryStore(dataset)
    if isinstance(dataset, (str, Path)):
        path = Path(dataset)
        if not path.exists():
            raise FileNotFoundError(f"No dataset file at {path}.")
        if path.suffix in NPY_SUFFIXES:
            return NpyStore(path)
        if path.suffix in HDF5_SUFFIXES:
            return HDF5Store(path, key=key)
        raise ValueError(
            f"Unsupported dataset format {path.suffix!r}; use one of "
            f"{NPY_SUFFIXES + HDF5_SUFFIXES}, or wrap other sources in CallableStore."
        )
    raise TypeError(
        "dataset must be a numpy array, a .npy/.h5/.hdf5 path, or a VectorStore; "
        f"got {type(dataset).__name__}."
    )
