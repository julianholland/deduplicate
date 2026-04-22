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

        self.set_dataset_array(dataset_array) if dataset_array.size > 0 else None
        self.preinitialize_dataset_array()
    # In DuplicateDetectionAlgorithm

    @property
    def dataset_array(self) -> np.ndarray:
        view = self._dataset_array.view()
        view.flags.writeable = False
        return view
    
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

    def compute_distance_matrix(self, vector_array: np.ndarray) -> None:
        """Compute the distance matrix for the dataset from scratch."""
        if self.distance_matrix.size == 0:
            self.initialize_distance_matrix()

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
        

    def pre_dda_processing(
        self, input_dataset_array: np.ndarray | None = None, *args, **kwargs
    ) -> None:
        """A method that can be overridden by child classes to perform any necessary processing before duplication checks. For example, this could be used to compute the distance matrix for the dataset before any duplication checks are performed, which would save time if multiple duplication checks are being performed on the same dataset with different input vectors."""
        pass

    def add_input_vector_to_dda(self) -> None:
        """Add the input vector to the dataset array and update the distance matrix accordingly."""
        pass

    def get_unique_vector_indices(self) -> np.ndarray:
        """Returns the indices of the unique vectors in the dataset."""
        if self.unique_vector_indices.shape[0] != self.vector_count:
            raise ValueError(
                "Unique vector indices array shape does not match dataset; please run get_dataset_unique_structures() to update the unique vector indices before calling this method."
            )
        return np.where(self.unique_vector_indices)[0]

    def deduplicate(self):
        """Finds all unique vectors in the dataset and returns them as a new array."""
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
        """Create new dataset array with correct shape and vector count"""
        if new_dataset_array.shape[0] > self.max_vector_array_size:
            raise ValueError("New dataset array size exceeds maximum allowed size.")
        self.vector_count = new_dataset_array.shape[0]
        self.initialize_dataset_array(new_dataset_array.shape[1])
        
        self._dataset_array[: self.vector_count] = new_dataset_array
        
    
    def preinitialize_dataset_array(self) -> None:
        has_input = self.input_vector.size > 0
        has_dataset = self._dataset_array.size > 0

        if has_input:
            vector_length = self.input_vector.shape[0]
        elif has_dataset:
            vector_length = self._dataset_array.shape[1]
        else:
            raise ValueError(
                "Cannot determine vector length from input vector or dataset array. "
                "Assign one of them before preinitialization."
            )

        if has_input and has_dataset and self._dataset_array.shape[1] != vector_length:
            raise ValueError("Dataset array vector length does not match input vector length.")

        if self._dataset_array.shape[0] > self.max_vector_array_size:
            raise ValueError("Dataset array size exceeds maximum allowed size.")

        if not has_dataset:
            self.initialize_dataset_array(vector_length)
            return

        if self._dataset_array.shape[0] != self.max_vector_array_size:
            existing_data = self._dataset_array.copy()
            self.vector_count = existing_data.shape[0]
            self.initialize_dataset_array(vector_length)
            self._dataset_array[: self.vector_count] = existing_data

    @abstractmethod
    def duplicate_check(self) -> bool:
        pass # pragma: no cover

    @abstractmethod
    def get_dataset_unique_structures(self) -> int:
        pass # pragma: no cover
