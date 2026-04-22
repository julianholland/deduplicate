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
        max_vector_array_size: int = 10000,
    ) -> None:
        super().__init__(
            tolerance, input_vector, dataset_array, distance_matrix, distance_metric, unique_vector_indices, max_vector_array_size
        )

    def __str__(self) -> str:
        return f"DistanceMatrix(tolerance={self.tolerance}, distance_metric={self.distance_metric})"

    def duplicate_check(self) -> bool:
        return bool(
            np.any(
                self.get_new_distance_matrix_column(self.dataset_array) < self.tolerance
            )
        )

    def get_dataset_unique_structures(self) -> int:
        self.unique_vector_indices = np.zeros(self.vector_count, dtype=bool)
        self.unique_vector_indices[0] = True  # the first vector is always unique
        for i in range(1, self.vector_count):
            imask = np.arange(self.vector_count) != i
            if np.all(self.distance_matrix[i, : self.vector_count][imask] >= self.tolerance):
                self.unique_vector_indices[i] = True
        return np.sum(self.unique_vector_indices)

    def pre_dda_processing(self, *args, **kwargs) -> None:
        self.preinitialize_dataset_array()
        self.initialize_distance_matrix()
        self.compute_distance_matrix(self.dataset_array)
    
    def add_input_vector_to_dda(self) -> None:
        """Add the input vector to the dataset array and update the distance matrix accordingly."""
        self.add_new_vector_to_distance_matrix(self.dataset_array)
        self._dataset_array[self.vector_count] = self.input_vector
        self.vector_count += 1
        