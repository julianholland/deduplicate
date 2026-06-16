import numpy as np
from abc import ABC, abstractmethod
from numba import njit


@njit
def fast_compute_distance_matrix(vector_array, distance_func):
    num_samples = vector_array.shape[0] # pragma: no cover, tested but does not appear in coverage report due to numba jit compilation
    distance_matrix = np.zeros((num_samples, num_samples)) # pragma: no cover
    for i in range(num_samples): # pragma: no cover
        for j in range(i + 1, num_samples): # pragma: no cover
            distance = distance_func(vector_array[i], vector_array[j]) # pragma: no cover
            distance_matrix[i, j] = distance # pragma: no cover
            distance_matrix[j, i] = distance # pragma: no cover
    return distance_matrix # pragma: no cover


@njit
def fast_get_new_distance_matrix_column(
    input_vector: np.ndarray, vector_array: np.ndarray, distance_func
) -> np.ndarray:
    """Calculates the distance matrix for the distances of a new input vector.

    Returns:
        np.ndarray: A 1D array containing the distances from the input vector to each vector in the dataset.
    """
    num_samples = vector_array.shape[0] # pragma: no cover
    new_distances = np.zeros(num_samples) # pragma: no cover
    for i in range(num_samples): # pragma: no cover
        new_distances[i] = distance_func(input_vector, vector_array[i]) # pragma: no cover
    return new_distances # pragma: no cover


# must be jit compatible functions, so defined outside of the class and not as static methods
@njit
def euclidean_distance(v1, v2):
    return float(np.linalg.norm(v1 - v2)) # pragma: no cover


@njit
def manhattan_distance(v1, v2):
    return np.sum(np.abs(v1 - v2)) # pragma: no cover


@njit
def cosine_distance(v1, v2):
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))  # pragma: no cover


@njit
def hamming_distance(v1, v2):
    return np.sum(v1 != v2) # pragma: no cover


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
    ) -> None:
        self.tolerance = tolerance
        self.input_vector = input_vector
        self._dataset_array = np.array([])  # will be initialized properly in preinitialize_dataset_array
        
        self.distance_matrix = distance_matrix
        self.distance_metric = distance_metric
        self.unique_vector_indices = unique_vector_indices
        self.max_vector_array_size = max_vector_array_size
        self._dirty = True  # auxiliary structures (distance matrix, hash dict, ...) need (re)building
        self.vector_count = 0

        self.set_dataset_array(dataset_array) if dataset_array.size > 0 else None

    @property
    def dataset_array(self) -> np.ndarray:
        view = self._dataset_array.view()
        view.flags.writeable = False
        return view
    
    @dataset_array.setter
    def dataset_array(self, _value: np.ndarray) -> None:
        raise AttributeError(
            "Cannot assign to 'dataset_array' directly; use set_dataset_array(...)."
        )
    
    def _set_dataset_array_internal(self, arr:np.ndarray) -> None:
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

        self.distance_matrix[: self.vector_count, : self.vector_count] = fast_compute_distance_matrix(
            vector_array[: self.vector_count], self.distance_function
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
        self.distance_matrix[self.vector_count] = np.pad(new_distances, (0, self.max_vector_array_size - len(new_distances)), constant_values=0)
        self.distance_matrix[:, self.vector_count] = np.pad(new_distances, (0, self.max_vector_array_size - len(new_distances)), constant_values=0)
        

    def pre_dda_processing(self, *args, **kwargs) -> None:
        """Ensure the dataset array and any algorithm-specific auxiliary structures (e.g. distance matrix, hash dictionary) are preallocated and up to date before duplication checks.

        This is idempotent: auxiliary structures are only rebuilt via `_rebuild_auxiliary_structures()` if the dataset has changed (`_dirty`) since they were last built. Child classes should implement `_rebuild_auxiliary_structures()` rather than overriding this method.
        """
        self.preinitialize_dataset_array()
        if self._dirty:
            self._rebuild_auxiliary_structures()
            self._dirty = False

    def _rebuild_auxiliary_structures(self) -> None:
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

    def _append_vector_to_structures(self) -> None:
        """Append `input_vector` to the dataset array and incrementally update any auxiliary structures. Overridden by child classes."""
        pass

    def get_unique_vector_indices(self) -> np.ndarray:
        """Returns the indices of the unique vectors in the dataset."""
        if self.unique_vector_indices.shape[0] != self.vector_count:
            raise ValueError(
                "Unique vector indices array shape does not match dataset; please run get_dataset_unique_structures() to update the unique vector indices before calling this method."
            )
        return np.where(self.unique_vector_indices)[0]

    def deduplicate(self):
        """Return all unique vectors in the dataset as a new array.

        Returns
        -------
        np.ndarray
            2-D array whose rows are the unique vectors.

        Notes
        -----
        Calls ``get_dataset_unique_structures()`` (via ``pre_dda_processing``) to
        ensure auxiliary structures are up to date before selecting unique rows.
        """
        return self.dataset_array[self.get_unique_vector_indices()]
    
    def initialize_dataset_array(self, vector_length: int) -> None:
        self._set_dataset_array_internal(np.zeros((self.max_vector_array_size, vector_length)))

    def initialize_distance_matrix(self) -> None:
        self.distance_matrix = np.zeros((self.max_vector_array_size, self.max_vector_array_size))
    
    def get_filled_dataset_array(self) -> np.ndarray:
        return self._dataset_array[: self.vector_count]
    
    def get_filled_distance_matrix(self) -> np.ndarray:
        return self.distance_matrix[: self.vector_count, : self.vector_count]

    def set_dataset_array(self, new_dataset_array: np.ndarray) -> None:
        """Create new dataset array with correct shape and vector count
        if the input vector and dataset are both empty then the vector length cannot be determined, so we initialize an empty dataset array"""
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
        has_input = self.input_vector.size > 0
        has_dataset = self._dataset_array.size > 0

        if has_input and has_dataset:
            print(self.input_vector.shape, self._dataset_array.shape)
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
        vector_length=self.get_vector_length()
        
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
        pass # pragma: no cover

    @abstractmethod
    def get_dataset_unique_structures(self) -> int:
        pass # pragma: no cover
