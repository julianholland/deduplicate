from deduplicate_lib.core.tolerance_calculator import ToleranceCalculator
from deduplicate_lib.core.duplicate_detection_algorithm import DuplicateDetectionAlgorithm
from deduplicate_lib.core.plugin_registry import register_plugin
import numpy as np


@register_plugin("tolerance_calculator", "perturbed_dataset_reclustering")
class PerturbedDatasetReclustering(ToleranceCalculator):
    """Tolerance calculator based on perturbed-dataset reclustering.

    Creates ``perturbations_per_vector`` noisy copies of each original vector,
    then binary-searches for the tolerance that recovers the original unique-vector
    count in the perturbed dataset.  The returned value depends on
    ``target_unique_vectors_threshold``: ``"loose"`` returns the highest valid
    tolerance, ``"tight"`` the lowest, and ``"average"`` their mean.

    Parameters
    ----------
    duplicate_detection_algorithm_object : DuplicateDetectionAlgorithm
        The DDA used to count unique structures during the search.
    tolerance_dataset_array : np.ndarray, optional
        Dataset to search over.  Defaults to the DDA's current dataset.
    perturbations_per_vector : int, optional
        Number of perturbed copies per original vector.  Defaults to ``1``
        (no perturbation; pass the original vectors directly).
    perturbation_scale : float, optional
        Standard deviation of the Gaussian noise added per perturbation.
        Defaults to ``0.1``.
    binary_search_steps : int, optional
        Number of binary-search iterations.  Defaults to ``20``.
    target_unique_vectors : int or None, optional
        Target unique-vector count.  ``None`` uses the current dataset size.
    target_unique_vectors_threshold : str, optional
        One of ``"average"``, ``"loose"``, or ``"tight"``.  Defaults to
        ``"average"``.
    """
    def __init__(self,
        duplicate_detection_algorithm_object: DuplicateDetectionAlgorithm,
        tolerance_dataset_array: np.ndarray = np.array([]),
        perturbations_per_vector: int = 1,
        perturbation_scale: float = 0.1,
        binary_search_steps: int = 20,
        target_unique_vectors: int | None = None,
        target_unique_vectors_threshold: str = "average"
    ):

        super().__init__(
            duplicate_detection_algorithm_object=duplicate_detection_algorithm_object,
            tolerance_dataset_array=tolerance_dataset_array,
            perturbations_per_vector=perturbations_per_vector,
            perturbation_scale=perturbation_scale,
            binary_search_steps=binary_search_steps,
        )

        if target_unique_vectors is None:
            self.target_unique_vectors = duplicate_detection_algorithm_object.vector_count
        else:
            self.target_unique_vectors = target_unique_vectors
        
        self.target_unique_vectors_threshold = target_unique_vectors_threshold
    def __str__(self) -> str:
        return f"PerturbedDatasetReclustering(perturbations_per_vector={self.perturbations_per_vector}, perturbation_scale={self.perturbation_scale}, dda={str(self.duplicate_detection_algorithm_object).split('(')[0]})"
    
    def calculate_tolerance(self) -> float:
        """Find the highest and lowest values of tolerance that yield the length of the original dataset from the perturbed dataset, then returns the average of those two values.
        
        Returns:
            float: The calculated tolerance value.
        """
        self.duplicate_detection_algorithm_object.pre_dda_processing()
        print(self.duplicate_detection_algorithm_object.vector_count)
        if self.target_unique_vectors_threshold == "average":        
            low_tolerance = self.binary_search_tolerance(
                target_unique_vectors=self.target_unique_vectors,
                find_largest_tolerance_for_target=False,
            )
            high_tolerance = self.binary_search_tolerance(
                target_unique_vectors=self.target_unique_vectors,
                find_largest_tolerance_for_target=True,
            )
            return (low_tolerance + high_tolerance) / 2
        
        elif self.target_unique_vectors_threshold == "loose":
            return self.binary_search_tolerance(
                target_unique_vectors=self.target_unique_vectors,
                find_largest_tolerance_for_target=True,
            )
        elif self.target_unique_vectors_threshold == "tight":
            return self.binary_search_tolerance(
                target_unique_vectors=self.target_unique_vectors,
                find_largest_tolerance_for_target=False,
            )
        else:
            raise ValueError(f"Invalid target_unique_vectors_threshold: {self.target_unique_vectors_threshold}. Must be 'average', 'loose', or 'tight'.")