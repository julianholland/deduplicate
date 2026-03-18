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

    def duplicate_check(self) -> np.bool_:
        return np.any(self.get_new_distance_matrix_column() < self.tolerance)

    def get_dataset_unique_structures(self) -> int:
        unique_structures = 0
        for i in range(self.distance_matrix.shape[0]):
            if np.all(self.distance_matrix[i][not i] >= self.tolerance):
                unique_structures += 1
        return unique_structures
