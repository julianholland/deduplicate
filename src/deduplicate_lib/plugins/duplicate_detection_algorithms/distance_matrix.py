from deduplicate_lib.core.duplicate_detection_algorithm import (
    DuplicateDetectionAlgorithm,
)
from deduplicate_lib.core.plugin_registry import register_plugin
from deduplicate_lib.utils.array_manager import DataArray
import numpy as np
import warnings


@register_plugin(kind="duplicate_detection_algorithm", name="distance_matrix")
class DistanceMatrix(DuplicateDetectionAlgorithm):
    def __init__(
        self,
        tolerance: float = 0.1,
        input_vector: np.ndarray = np.array([]),
        dataset_array: np.ndarray = np.array([]),
        distance_matrix: np.ndarray = np.array([]),
        distance_vector: np.ndarray = np.array([]),
        distance_metric: str = "euclidean",
        unique_vector_indices: np.ndarray = np.array([]),
        max_dataset_size: int = 10000,
    ) -> None:
        super().__init__(
            tolerance,
            input_vector,
            dataset_array,
            distance_matrix,
            distance_vector,
            distance_metric,
            unique_vector_indices,
            max_dataset_size,
        )

    def __str__(self) -> str:
        return f"DistanceMatrix(tolerance={self.tolerance}, distance_metric={self.distance_metric})"

    def _ensure_distance_matrix(self):
        self.distance_matrix = self.get_distance_matrix()
        if self.distance_matrix.shape[0] != self.dataset_array.shape[0]:
            warnings.warn("Distance matrix shape does not match dataset; recomputing.")
            self.compute_distance_matrix(self.dataset_array)

    def duplicate_check(self) -> bool:
        self.update_distance_vector(self._dataset_array.data_array[:self.dataset_vector_number])
        return bool(
            np.any(self.distance_vector < self.tolerance)
            )

    def get_dataset_unique_structures(self) -> int:
        self._ensure_distance_matrix()
        self.unique_vector_indices = np.zeros(self.dataset_vector_number, dtype=bool)
        self.unique_vector_indices[0] = True  # the first vector is always unique
        self.distance_matrix=self.get_distance_matrix()
        for i in range(1, self.dataset_vector_number):
            imask = np.arange(self.dataset_vector_number) != i
            if np.all(self.distance_matrix[i][imask] >= self.tolerance):
                self.unique_vector_indices[i] = True
        return np.sum(self.unique_vector_indices)

    def pre_dda_processing(
        self, *args, **kwargs
    ) -> None:
        # set up quick add _dataset_array so we can add the input vector to it and update the distance matrix without needing to resize the array which can be very slow
        self._dataset_array = DataArray([self.dataset_array.shape[1]], [self.max_dataset_size], self.dataset_array, ['variable_0', 'fixed_0'])
        self.compute_distance_matrix(self._dataset_array.data_array[:self.dataset_vector_number])

    def add_input_vector_to_dda(self) -> None:
        """Add the input vector to the dataset array and update the distance matrix accordingly."""
        self._dataset_array.add_input_vector_to_data_array(self.input_vector, (self.dataset_vector_number,))
        self.add_new_vector_to_distance_matrix(self.input_vector)
        self.dataset_vector_number += 1
