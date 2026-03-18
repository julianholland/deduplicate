import numpy as np
from abc import ABC, abstractmethod


class DuplicateDetectionAlgorithm(ABC):
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

    def calculate_distance(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        if self.distance_metric == "euclidean":
            return float(np.linalg.norm(vector1 - vector2))
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(vector1 - vector2))
        elif self.distance_metric == "cosine":
            return 1 - np.dot(vector1, vector2) / (
                np.linalg.norm(vector1) * np.linalg.norm(vector2)
            )
        elif self.distance_metric == "hamming":
            return np.sum(vector1 != vector2)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def compute_distance_matrix(self) -> None:
        """Compute the distance matrix for the dataset from scratch."""
        num_samples = self.dataset_array.shape[0]
        self.distance_matrix = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                distance = self.calculate_distance(
                    self.dataset_array[i], self.dataset_array[j]
                )
                self.distance_matrix[i, j] = distance
                self.distance_matrix[j, i] = distance

    def get_new_distance_matrix_column(self) -> np.ndarray:
        """Calculates the distance matrix for the distances of a new input vector.

        Returns:
            np.ndarray: A 1D array containing the distances from the input vector to each vector in the dataset.
        """
        num_samples = self.dataset_array.shape[0]
        new_distances = np.zeros(num_samples)
        for i in range(num_samples):
            new_distances[i] = self.calculate_distance(
                self.input_vector, self.dataset_array[i]
            )
        return new_distances

    def add_new_vector_to_distance_matrix(self) -> None:
        """Add a new input vector to the distance matrix."""
        new_distances = self.get_new_distance_matrix_column()
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
