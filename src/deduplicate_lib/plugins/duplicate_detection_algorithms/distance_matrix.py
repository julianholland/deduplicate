from deduplicate_lib.core.duplicate_detection_algorithm import DuplicateDetectionAlgorithm
from deduplicate_lib.core.plugin_registry import register_plugin
import numpy as np

@register_plugin(kind="duplicate_detection_algorithm", name="distance_matrix")
class DistanceMatrix(DuplicateDetectionAlgorithm):
    """Exact deduplication using a full pairwise distance matrix.

    Computes all N×N pairwise distances.  Vectors are taken in dataset order:
    each one is kept as unique unless it is within ``tolerance`` of a vector
    already kept, so every group of near-duplicates keeps exactly one vector
    (its earliest) and the rest are flagged.  O(N²) time and space;
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
        """Keep the first vector of each group of near-duplicates.

        A vector is unique unless it is within ``tolerance`` of a vector
        already kept (earlier in the dataset). Comparing only against kept
        vectors means a group of mutual duplicates keeps one representative,
        rather than every member flagging every other.
        """
        self.pre_dda_processing()
        n = self.vector_count
        distances = self.distance_matrix[:n, :n]
        unique = np.zeros(n, dtype=bool)
        for i in range(n):
            unique[i] = not np.any(distances[i, :i][unique[:i]] < self.tolerance)
        self.unique_vector_indices = unique
        return int(np.sum(unique))

    def _rebuild_auxiliary_structures(self) -> None:
        self.compute_distance_matrix(self.dataset_array)

    def _append_vector_to_structures(self) -> None:
        """Add the input vector to the dataset array and update the distance matrix accordingly."""
        self.add_new_vector_to_distance_matrix(self.dataset_array)
        self._dataset_array[self.vector_count] = self.input_vector
        self.vector_count += 1
