from deduplicate.core.duplicate_detection_algorithm import DuplicateDetectionAlgorithm
from deduplicate.core.plugin_registry import register_plugin
import numpy as np


@register_plugin(kind="duplicate_detection_algorithm", name="distance_matrix")
class DistanceMatrix(DuplicateDetectionAlgorithm):
    def __init__(
        self,
        tolerance: float,
        input_vector: np.ndarray,
        dataset_array: np.ndarray,
        distance_matrix: np.ndarray = np.array([]),
        distance_metric: str = "euclidean",
    ) -> None:
        super().__init__(
            tolerance, input_vector, dataset_array, distance_matrix, distance_metric
        )

    def __str__(self) -> str:
        return f"DistanceMatrix(tolerance={self.tolerance}, distance_metric={self.distance_metric})"

    def _ensure_distance_matrix(self):
        if self.distance_matrix.shape[0] != self.dataset_array.shape[0]:
            # warnings.warn(f"Distance matrix shape does not match dataset; recomputing.{self.dataset_array.shape[0]}, {self.distance_matrix.shape[0]}")
            # self.compute_distance_matrix(self.dataset_array)
            raise ValueError("Distance matrix shape does not match dataset; recomputing is currently disabled to avoid long computations during testing. Please compute the distance matrix manually and assign it to the duplicate detection algorithm object before duplication checks.")

    def duplicate_check(self) -> bool:
        return bool(
            np.any(
                self.get_new_distance_matrix_column(self.dataset_array) < self.tolerance
            )
        )

    def get_dataset_unique_structures(self) -> int:
        self._ensure_distance_matrix()
        unique_structures = 1
        
        for i in range(1, self.distance_matrix.shape[0]):
            imask = np.arange(self.distance_matrix.shape[0]) != i
            if np.all(self.distance_matrix[i][imask] >= self.tolerance):
                unique_structures += 1
        return unique_structures

    def pre_dda_processing(self) -> None:
        self.compute_distance_matrix(self.dataset_array)