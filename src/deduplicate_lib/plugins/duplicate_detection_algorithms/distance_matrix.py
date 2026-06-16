from deduplicate_lib.core.duplicate_detection_algorithm import DuplicateDetectionAlgorithm
from deduplicate_lib.core.plugin_registry import register_plugin
import numpy as np

@register_plugin(kind="duplicate_detection_algorithm", name="distance_matrix")
class DistanceMatrix(DuplicateDetectionAlgorithm):
    """Exact deduplication using a full pairwise distance matrix.

    Computes all N×N pairwise distances and flags any vector whose nearest
    neighbour is within ``tolerance`` as a duplicate.  O(N²) time and space;
    suitable for small-to-medium datasets (up to ~5 000 vectors) where exact
    results are required.  Supports all four distance metrics: ``"euclidean"``,
    ``"manhattan"``, ``"cosine"``, and ``"hamming"``.

    Parameters
    ----------
    tolerance : float, optional
        Distance threshold for duplicate detection.  Defaults to ``0.1``.
    input_vector : np.ndarray, optional
        Single vector to check against the dataset.
    dataset_array : np.ndarray, optional
        Initial dataset of vectors (rows).
    distance_matrix : np.ndarray, optional
        Pre-computed distance matrix.  Allocated automatically if empty.
    distance_metric : str, optional
        One of ``"euclidean"``, ``"manhattan"``, ``"cosine"``, ``"hamming"``.
        Defaults to ``"euclidean"``.
    unique_vector_indices : np.ndarray, optional
        Boolean array marking unique vectors.
    max_vector_array_size : int, optional
        Maximum number of vectors the pre-allocated arrays can hold.
        Defaults to 10000.
    """

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
        self.preinitialize_dataset_array()
        return bool(
            np.any(
                self.get_new_distance_matrix_column(self.dataset_array) < self.tolerance
            )
        )

    def get_dataset_unique_structures(self) -> int:
        self.pre_dda_processing()
        self.unique_vector_indices = np.zeros(self.vector_count, dtype=bool)
        self.unique_vector_indices[0] = True  # the first vector is always unique
        for i in range(1, self.vector_count):
            imask = np.arange(self.vector_count) != i
            if np.all(self.distance_matrix[i, : self.vector_count][imask] >= self.tolerance):
                self.unique_vector_indices[i] = True
        return np.sum(self.unique_vector_indices)

    def _rebuild_auxiliary_structures(self) -> None:
        self.compute_distance_matrix(self.dataset_array)

    def _append_vector_to_structures(self) -> None:
        """Add the input vector to the dataset array and update the distance matrix accordingly."""
        self.add_new_vector_to_distance_matrix(self.dataset_array)
        self._dataset_array[self.vector_count] = self.input_vector
        self.vector_count += 1
