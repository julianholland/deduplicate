import numpy as np
from abc import ABC, abstractmethod

def euclidean_distance(v1, v2):
    return float(np.linalg.norm(v1 - v2))

def manhattan_distance(v1, v2):
    return np.sum(np.abs(v1 - v2))

def cosine_distance(v1, v2):
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def hamming_distance(v1, v2):
    return np.sum(v1 != v2)

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
        input_vector: np.ndarray,
        dataset_array: np.ndarray,
        distance_matrix: np.ndarray = np.array([]),
        distance_metric: str = "euclidean",
    ) -> None:
        self.tolerance = tolerance
        self.input_vector = input_vector
        self.dataset_array = dataset_array
        self.distance_matrix = distance_matrix
        self.distance_metric = distance_metric

    # In DuplicateDetectionAlgorithm

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
        num_samples = vector_array.shape[0]
        self.distance_matrix = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                distance = self.calculate_distance(
                    vector_array[i], vector_array[j]
                )
                self.distance_matrix[i, j] = distance
                self.distance_matrix[j, i] = distance

    def get_new_distance_matrix_column(self, vector_array: np.ndarray) -> np.ndarray:
        """Calculates the distance matrix for the distances of a new input vector.

        Returns:
            np.ndarray: A 1D array containing the distances from the input vector to each vector in the dataset.
        """
        num_samples = vector_array.shape[0]
        new_distances = np.zeros(num_samples)
        for i in range(num_samples):
            new_distances[i] = self.calculate_distance(
                self.input_vector, vector_array[i]
            )
        return new_distances

    def add_new_vector_to_distance_matrix(self, vector_array: np.ndarray) -> None:
        """Add a new input vector to the distance matrix."""
        new_distances = self.get_new_distance_matrix_column(vector_array)
        self.distance_matrix = np.hstack(
            (self.distance_matrix, new_distances[:, np.newaxis])
        )
        new_row = np.append(new_distances, 0)  # Distance to itself is 0
        self.distance_matrix = np.vstack((self.distance_matrix, new_row))

    @abstractmethod
    def duplicate_check(self) -> bool:
        pass

    @abstractmethod
    def get_dataset_unique_structures(self) -> int:
        pass
