from deduplicate.core.duplicate_detection_algorithm import DuplicateDetectionAlgorithm
from deduplicate.core.plugin_registry import register_plugin
import numpy as np
import warnings

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
            warnings.warn("Distance matrix shape does not match dataset; recomputing.")
            self.compute_distance_matrix(self.dataset_array)
            

    def duplicate_check(self) -> bool:
        return bool(np.any(self.get_new_distance_matrix_column(self.dataset_array) < self.tolerance))

    def get_dataset_unique_structures(self) -> int:
        self._ensure_distance_matrix()
        unique_structures = 0
        for i in range(self.distance_matrix.shape[0]):
            if np.all(self.distance_matrix[i][not i] >= self.tolerance):
                unique_structures += 1
        print(self.distance_matrix)
        return unique_structures

    