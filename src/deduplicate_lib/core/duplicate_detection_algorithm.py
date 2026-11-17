import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from numba import njit

from deduplicate_lib.core import resources
from deduplicate_lib.core.vector_store import InMemoryStore, VectorStore, open_store

logger = logging.getLogger(__name__)


@njit
def fast_compute_distance_matrix(vector_array, distance_func):
    num_samples = vector_array.shape[
        0
    ]  # pragma: no cover, tested but does not appear in coverage report due to numba jit compilation
    distance_matrix = np.zeros((num_samples, num_samples))  # pragma: no cover
    for i in range(num_samples):  # pragma: no cover
        for j in range(i + 1, num_samples):  # pragma: no cover
            distance = distance_func(
                vector_array[i], vector_array[j]
            )  # pragma: no cover
            distance_matrix[i, j] = distance  # pragma: no cover
            distance_matrix[j, i] = distance  # pragma: no cover
    return distance_matrix  # pragma: no cover


@njit
def fast_get_new_distance_matrix_column(
    input_vector: np.ndarray, vector_array: np.ndarray, distance_func
) -> np.ndarray:
    """Calculates the distance matrix for the distances of a new input vector.

    Returns:
        np.ndarray: A 1D array containing the distances from the input vector to each vector in the dataset.
    """
    num_samples = vector_array.shape[0]  # pragma: no cover
    new_distances = np.zeros(num_samples)  # pragma: no cover
    for i in range(num_samples):  # pragma: no cover
        new_distances[i] = distance_func(
            input_vector, vector_array[i]
        )  # pragma: no cover
    return new_distances  # pragma: no cover


# must be jit compatible functions, so defined outside of the class and not as static methods
@njit
def euclidean_distance(v1, v2):
    return float(np.linalg.norm(v1 - v2))  # pragma: no cover


@njit
def manhattan_distance(v1, v2):
    return np.sum(np.abs(v1 - v2))  # pragma: no cover


@njit
def cosine_distance(v1, v2):
    return 1 - np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )  # pragma: no cover


@njit
def hamming_distance(v1, v2):
    return np.sum(v1 != v2)  # pragma: no cover


DISTANCE_FUNCTIONS = {
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
    "cosine": cosine_distance,
    "hamming": hamming_distance,
}


class DuplicateDetectionAlgorithm(ABC):
    """Abstract base class for deduplication algorithms operating on numpy vector arrays.

    Subclasses implement ``duplicate_check()`` and ``get_dataset_unique_structures()``
    using algorithm-specific auxiliary structures (e.g. a distance matrix or hash
    dictionary).  The public interface is:

    * ``tolerance`` — distance threshold below which two vectors are considered duplicates.
    * ``input_vector`` — the single vector to check or append.
    * ``dataset_array`` — read-only view of the currently loaded vectors.
    * ``vector_count`` — number of vectors currently in the dataset.

    ``_dataset_array`` and ``distance_matrix`` are pre-allocated to
    ``max_vector_array_size`` rows to avoid repeated reallocation.  Load data via
    ``set_dataset_array()``; never assign to ``dataset_array`` directly.

    Datasets too large for memory can instead be given as ``dataset``: a
    ``.npy``/``.h5`` path or a :class:`~deduplicate_lib.core.vector_store.VectorStore`
    (e.g. a ``CallableStore`` over a database).  They are then read in chunks
    sized by ``memory_limit`` and never loaded whole; ``dataset_array`` is not
    available for them, and algorithms that need the whole dataset in memory
    (``DistanceMatrix``) refuse them.

    Parameters
    ----------
    tolerance : float
        Distance threshold for duplicate detection.
    input_vector : np.ndarray, optional
        Single vector to check against the dataset.
    dataset_array : np.ndarray, optional
        Initial dataset of vectors (rows).  Copied into the pre-allocated backing
        array.
    distance_matrix : np.ndarray, optional
        Pre-computed distance matrix.  If empty, it is allocated on first use.
    distance_metric : str, optional
        One of ``"euclidean"``, ``"manhattan"``, ``"cosine"``, ``"hamming"``.
        Defaults to ``"euclidean"``.
    unique_vector_indices : np.ndarray, optional
        Boolean array marking unique vectors.  Updated by
        ``get_dataset_unique_structures()``.
    max_vector_array_size : int, optional
        Maximum number of vectors the pre-allocated arrays can hold.
        Defaults to 10000.
    dataset : str, Path or VectorStore, optional
        On-disk (or otherwise out-of-core) dataset, instead of ``dataset_array``.
        A numpy array is accepted too and treated like ``dataset_array``.
    memory_limit : int, str or None, optional
        Memory budget for chunked processing, in bytes or as a string such as
        ``"8GB"``.  ``"auto"`` (default) uses half of the available memory.
        Also the size above which ``deduplicate()`` writes its result to a file
        instead of returning it.
    n_threads : int, optional
        Threads used by parallel kernels.  ``None`` (default) keeps numba's setting.
    progress : bool, optional
        Show progress bars for long computations (needs ``tqdm``).
    """

    ALLOWED_DISTANCES = DISTANCE_FUNCTIONS

    def __init__(
        self,
        tolerance: float,
        input_vector: np.ndarray = np.array([]),
        dataset_array: np.ndarray = np.array([]),
        distance_matrix: np.ndarray = np.array([]),
        distance_metric: str = "euclidean",
        unique_vector_indices: np.ndarray = np.array([]),
        max_vector_array_size: int = 10000,
        dataset: str | Path | VectorStore | np.ndarray | None = None,
        memory_limit: int | float | str | None = "auto",
        n_threads: int | None = None,
        progress: bool = False,
    ) -> None:
        self.tolerance = tolerance
        self.input_vector = input_vector
        self._dataset_array = np.array(
            []
        )  # will be initialized properly in preinitialize_dataset_array

        self.distance_matrix = distance_matrix
        self.distance_metric = distance_metric
        self.unique_vector_indices = unique_vector_indices
        self.max_vector_array_size = max_vector_array_size
        self._dirty = True  # auxiliary structures (distance matrix, hash dict, ...) need (re)building
        self.vector_count = 0
        self._store: VectorStore | None = None  # set for out-of-core datasets only
        # whether the limit was derived from the free memory ("auto") rather
        # than given: algorithms only refuse work over an explicit limit
        self.memory_limit_is_auto = memory_limit is None or (
            isinstance(memory_limit, str) and memory_limit.strip().lower() == "auto"
        )
        self.memory_limit = resources.parse_memory_limit(memory_limit)
        if n_threads is not None and (
            not isinstance(n_threads, int)
            or isinstance(n_threads, bool)
            or n_threads < 1
        ):
            raise ValueError("n_threads must be a positive integer or None.")
        self.n_threads = n_threads
        self.progress = progress

        if dataset is not None and dataset_array.size > 0:
            raise ValueError("Pass either dataset_array or dataset, not both.")
        if isinstance(dataset, np.ndarray):
            dataset_array = dataset
            dataset = None

        self.set_dataset_array(dataset_array) if dataset_array.size > 0 else None
        if dataset is not None:
            self.set_dataset_store(dataset)

    @property
    def is_out_of_core(self) -> bool:
        """Whether the dataset is read from a store in chunks rather than held in memory."""
        return self._store is not None

    @property
    def store(self) -> VectorStore:
        """The dataset as a :class:`VectorStore` (an in-memory view for in-memory datasets)."""
        if self._store is not None:
            return self._store
        if self._dataset_array.ndim != 2:
            return InMemoryStore(np.empty((0, self.get_vector_length())))
        return InMemoryStore(self._dataset_array[: self.vector_count])

    def set_dataset_store(self, dataset, key: str = "vectors") -> None:
        """Use an out-of-core dataset: a ``.npy``/``.h5`` path or a VectorStore."""
        store = open_store(dataset, key=key)
        if store.in_memory:
            self.set_dataset_array(np.asarray(store.read(0, len(store))))
            return
        self._store = store
        self._set_dataset_array_internal(np.array([]))
        self.vector_count = len(store)
        self.unique_vector_indices = np.array([])
        self._dirty = True
        logger.info("Using out-of-core dataset %r.", store)

    def _init_kwargs(self) -> dict:
        """Constructor arguments that reproduce this DDA's settings (not its data).

        Subclasses with extra settings extend this via ``super()``.
        """
        return {
            "tolerance": self.tolerance,
            "distance_metric": self.distance_metric,
            "memory_limit": "auto" if self.memory_limit_is_auto else self.memory_limit,
            "n_threads": self.n_threads,
            "progress": False,
        }

    def tuning_copy(self, array: np.ndarray) -> "DuplicateDetectionAlgorithm":
        """A new, independent in-memory DDA with these settings over ``array``.

        Tolerance calculators tune on such a copy, so the DDA they were given
        (its dataset, store, index and precomputed structures) is never touched.
        """
        array = np.asarray(array)
        return type(self)(
            **self._init_kwargs(),
            dataset_array=array,
            max_vector_array_size=max(len(array), 1),
        )

    def chunk_rows(self) -> int:
        """Rows per chunk when reading the dataset, from ``memory_limit``."""
        return resources.chunk_rows(self.store.dim, self.memory_limit)

    def _out_of_core_error(self, what: str) -> ValueError:
        return ValueError(
            f"{what} is not available: the dataset is out-of-core ({self._store!r}) "
            "and is never loaded into memory as a whole. Use deduplicate(out=...), "
            "get_unique_vector_mask(), or read rows with .store.read()/.store.take()."
        )

    @property
    def dataset_array(self) -> np.ndarray:
        if self._store is not None:
            raise self._out_of_core_error("dataset_array")
        view = self._dataset_array.view()
        view.flags.writeable = False
        return view

    @dataset_array.setter
    def dataset_array(self, _value: np.ndarray) -> None:
        raise AttributeError(
            "Cannot assign to 'dataset_array' directly; use set_dataset_array(...)."
        )

    def _set_dataset_array_internal(self, arr: np.ndarray) -> None:
        self._dataset_array = arr

    @property
    def distance_metric(self):
        return self._distance_metric

    @distance_metric.setter
    def distance_metric(self, value):
        if value not in self.ALLOWED_DISTANCES:
            raise ValueError(
                f"Unsupported distance metric: {value}, "
                f"supported metrics are: {list(self.ALLOWED_DISTANCES.keys())}"
            )
        self._distance_metric = value
        self.distance_function = self.ALLOWED_DISTANCES[value]

    def calculate_distance(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        return self.distance_function(vector1, vector2)

    def compute_distance_matrix(self, vector_array: np.ndarray | None = None) -> None:
        """Compute the distance matrix for the dataset from scratch."""
        if self.distance_matrix.size == 0:
            self.initialize_distance_matrix()

        if vector_array is None:
            vector_array = self._dataset_array

        self.distance_matrix[: self.vector_count, : self.vector_count] = (
            fast_compute_distance_matrix(
                vector_array[: self.vector_count], self.distance_function
            )
        )

    def get_new_distance_matrix_column(self, vector_array: np.ndarray) -> np.ndarray:
        """Calculates the distance matrix for the distances of a new input vector.

        Returns:
            np.ndarray: A 1D array containing the distances from the input vector to each vector in the dataset.
        """
        return fast_get_new_distance_matrix_column(
            self.input_vector, vector_array[: self.vector_count], self.distance_function
        )

    def add_new_vector_to_distance_matrix(self, vector_array: np.ndarray) -> None:
        """Add a new input vector to the distance matrix."""
        new_distances = self.get_new_distance_matrix_column(vector_array)
        self.distance_matrix[self.vector_count] = np.pad(
            new_distances,
            (0, self.max_vector_array_size - len(new_distances)),
            constant_values=0,
        )
        self.distance_matrix[:, self.vector_count] = np.pad(
            new_distances,
            (0, self.max_vector_array_size - len(new_distances)),
            constant_values=0,
        )

    def pre_dda_processing(self) -> None:
        """Ensure the dataset array and any algorithm-specific auxiliary structures (e.g. distance matrix, hash dictionary) are preallocated and up to date before duplication checks.

        This is idempotent: auxiliary structures are only rebuilt via `_rebuild_auxiliary_structures()` if the dataset has changed (`_dirty`) since they were last built. Child classes should implement `_rebuild_auxiliary_structures()` rather than overriding this method.
        """
        self.preinitialize_dataset_array()
        if self._dirty:
            self._rebuild_auxiliary_structures()
            self._dirty = False

    def _rebuild_auxiliary_structures(self) -> None:  # noqa: B027 (optional hook)
        """Rebuild any algorithm-specific auxiliary structures from the current dataset array. Overridden by child classes."""
        pass

    def add_input_vector_to_dda(self) -> None:
        """Add ``self.input_vector`` to the dataset and incrementally update auxiliary structures.

        ``self.input_vector`` must be set before calling.  Auxiliary structures
        (distance-matrix column or hash-dict entry) are updated incrementally
        rather than rebuilt from scratch, making this efficient for streaming
        ingestion.
        """
        self.pre_dda_processing()
        self._append_vector_to_structures()

    def _append_vector_to_structures(self) -> None:  # noqa: B027 (optional hook)
        """Append `input_vector` to the dataset array and incrementally update any auxiliary structures. Overridden by child classes."""
        pass

    def get_unique_vector_mask(self) -> np.ndarray:
        """Returns a boolean mask over the dataset, ``True`` for unique vectors."""
        if self.unique_vector_indices.shape[0] != self.vector_count:
            raise ValueError(
                "Unique vector indices array shape does not match dataset; please run get_dataset_unique_structures() to update the unique vector indices before calling this method."
            )
        return np.asarray(self.unique_vector_indices, dtype=bool).copy()

    def get_unique_vector_indices(self) -> np.ndarray:
        """Returns the indices of the unique vectors in the dataset."""
        return np.where(self.get_unique_vector_mask())[0]

    def deduplicate(self, out: str | Path | None = None):
        """Return the unique vectors, or write them to a file.

        Parameters
        ----------
        out : str or Path, optional
            File to write the unique vectors to (``.npy``, ``.h5``/``.hdf5``).

        Returns
        -------
        np.ndarray or VectorStore
            For an in-memory dataset with no ``out``, a 2-D array whose rows are
            the unique vectors.  Otherwise the unique vectors are streamed to
            ``out`` and the written file is returned, opened as a store (its
            ``.path`` says where).  Out-of-core datasets default to
            ``<input>_unique.<ext>`` next to the input when ``out`` is not given.

        Notes
        -----
        Calls ``get_dataset_unique_structures()`` first, so the result always
        reflects the current dataset and ``tolerance``.
        """
        self.get_dataset_unique_structures()
        mask = self.get_unique_vector_mask()
        n_unique = int(mask.sum())
        result_bytes = n_unique * self.store.dim * self.store.dtype.itemsize

        if out is None and not self.is_out_of_core:
            if result_bytes > self.memory_limit:
                logger.warning(
                    "The %d unique vectors (%d bytes) exceed memory_limit (%d bytes); "
                    "returning them anyway. Pass deduplicate(out=...) to write them "
                    "to a file instead.",
                    n_unique,
                    result_bytes,
                    self.memory_limit,
                )
            return self.get_filled_dataset_array()[mask]

        if out is None:
            out = self.store.default_output_path()
        written = self.store.write_subset(mask, out, chunk_rows=self.chunk_rows())
        logger.info("Wrote %d unique vectors to %s.", n_unique, written.path)
        return written

    def initialize_dataset_array(self, vector_length: int) -> None:
        self._set_dataset_array_internal(
            np.zeros((self.max_vector_array_size, vector_length))
        )

    def initialize_distance_matrix(self) -> None:
        self.distance_matrix = np.zeros(
            (self.max_vector_array_size, self.max_vector_array_size)
        )

    def get_filled_dataset_array(self) -> np.ndarray:
        if self._store is not None:
            raise self._out_of_core_error("get_filled_dataset_array()")
        return self._dataset_array[: self.vector_count]

    def get_filled_distance_matrix(self) -> np.ndarray:
        return self.distance_matrix[: self.vector_count, : self.vector_count]

    def set_dataset_array(self, new_dataset_array: np.ndarray) -> None:
        """Create new dataset array with correct shape and vector count
        if the input vector and dataset are both empty then the vector length cannot be determined, so we initialize an empty dataset array

        Replaces an out-of-core dataset, if one was set, with the in-memory array."""
        if self._store is not None:
            self._store = None
            self.vector_count = 0
            self._dirty = True
        current_dataset_array = self._dataset_array[: self.vector_count]
        if new_dataset_array.shape == current_dataset_array.shape and np.array_equal(
            new_dataset_array, current_dataset_array
        ):
            return

        if new_dataset_array.shape[0] > self.max_vector_array_size:
            raise ValueError("New dataset array size exceeds maximum allowed size.")

        self.vector_count = new_dataset_array.shape[0]
        if new_dataset_array.size == 0 and self.input_vector.size > 0:
            self.initialize_dataset_array(len(self.input_vector))
        elif new_dataset_array.size > 0:
            self.initialize_dataset_array(new_dataset_array.shape[1])
            self._dataset_array[: self.vector_count] = new_dataset_array
        else:
            self._dataset_array = new_dataset_array

        self._dirty = True

    def get_vector_length(self) -> int:
        if self._store is not None:
            if (
                self.input_vector.size > 0
                and self.input_vector.shape[0] != self._store.dim
            ):
                raise ValueError(
                    "Input vector length does not match dataset vector length."
                )
            return self._store.dim

        has_input = self.input_vector.size > 0
        has_dataset = self._dataset_array.size > 0

        if has_input and has_dataset:
            if self.input_vector.shape[0] != self._dataset_array.shape[1]:
                raise ValueError(
                    "Input vector length does not match dataset vector length."
                )
            vector_length = self.input_vector.shape[0]
        elif has_dataset:
            vector_length = self._dataset_array.shape[1]
        elif has_input:
            vector_length = len(self.input_vector)
        else:
            raise ValueError(
                "Cannot determine vector length from input vector or dataset array. "
                "Assign one of them before preinitialization."
            )

        return vector_length

    def preinitialize_dataset_array(self) -> None:
        vector_length = self.get_vector_length()
        if self._store is not None:
            return  # out-of-core datasets are read in chunks, never preallocated

        if self._dataset_array.shape[0] > self.max_vector_array_size:
            raise ValueError("Dataset array size exceeds maximum allowed size.")

        if self._dataset_array.size == 0:
            self.initialize_dataset_array(vector_length)
            self.vector_count = 0

        if self._dataset_array.shape[0] != self.max_vector_array_size:
            existing_data = self._dataset_array.copy()
            self.vector_count = existing_data.shape[0]
            self.initialize_dataset_array(vector_length)
            self._dataset_array[: self.vector_count] = existing_data

    @abstractmethod
    def duplicate_check(self) -> bool:
        """Check whether ``self.input_vector`` is a duplicate of any vector in the dataset.

        ``self.input_vector`` must be set before calling.

        Returns
        -------
        bool
            ``True`` if ``self.input_vector`` is within ``self.tolerance`` of any
            existing dataset vector; ``False`` otherwise.
        """
        pass  # pragma: no cover

    @abstractmethod
    def get_dataset_unique_structures(self) -> int:
        """Mark the unique vectors of the dataset and return how many there are.

        Every group of near-duplicates keeps exactly one vector, its earliest
        in dataset order; the others are flagged. The result is stored in
        ``unique_vector_indices`` (see ``get_unique_vector_indices()``).
        """
