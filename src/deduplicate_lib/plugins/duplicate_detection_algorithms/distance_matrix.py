from deduplicate_lib.core.duplicate_detection_algorithm import DuplicateDetectionAlgorithm
from deduplicate_lib.core.plugin_registry import register_plugin
import numpy as np


@register_plugin(kind="duplicate_detection_algorithm", name="distance_matrix")
class DistanceMatrix(DuplicateDetectionAlgorithm):
    def __init__(
        self,
        tolerance: float = 0.1,
        input_vector: np.ndarray = np.array([]),
        dataset_array: np.ndarray = np.array([]),
        distance_matrix: np.ndarray = np.array([]),
        distance_metric: str = "euclidean",
        unique_vector_indices: np.ndarray = np.array([]),
    ) -> None:
        super().__init__(
            tolerance, input_vector, dataset_array, distance_matrix, distance_metric, unique_vector_indices
        )

    def __str__(self) -> str:
        return f"DistanceMatrix(tolerance={self.tolerance}, distance_metric={self.distance_metric})"

    def _ensure_distance_matrix(self):
        if self.distance_matrix.shape[0] != self.dataset_array.shape[0]:
            # warnings.warn(f"Distance matrix shape does not match dataset; recomputing.{self.dataset_array.shape[0]}, {self.distance_matrix.shape[0]}")
            # self.compute_distance_matrix(self.dataset_array)
            print(self.dataset_array.shape, self.distance_matrix.shape)
            raise ValueError("Distance matrix shape does not match dataset; recomputing is currently disabled to avoid long computations during testing. Please compute the distance matrix manually and assign it to the duplicate detection algorithm object before duplication checks.")

    def duplicate_check(self) -> bool:
        return bool(
            np.any(
                self.get_new_distance_matrix_column(self.dataset_array) < self.tolerance
            )
        )

    def get_dataset_unique_structures(self) -> int:
        self._ensure_distance_matrix()
        self.unique_vector_indices = np.zeros(self.dataset_array.shape[0], dtype=bool)
        self.unique_vector_indices[0] = True  # the first vector is always unique
        for i in range(1, self.distance_matrix.shape[0]):
            imask = np.arange(self.distance_matrix.shape[0]) != i
            if np.all(self.distance_matrix[i][imask] >= self.tolerance):
                self.unique_vector_indices[i] = True
        return np.sum(self.unique_vector_indices)

    def pre_dda_processing(self, input_dataset_array: np.ndarray | None = None, *args, **kwargs) -> None:
        if input_dataset_array is None:
            input_dataset_array = self.dataset_array
        self.compute_distance_matrix(input_dataset_array)
    
    def add_input_vector_to_dda(self) -> None:
        """Add the input vector to the dataset array and update the distance matrix accordingly."""
        self.add_new_vector_to_distance_matrix(self.dataset_array)
        self.dataset_array = np.vstack((self.dataset_array, self.input_vector))

