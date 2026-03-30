from deduplicate_lib.core.tolerance_calculator import ToleranceCalculator
from deduplicate_lib.core.duplicate_detection_algorithm import DuplicateDetectionAlgorithm
from deduplicate_lib.core.plugin_registry import register_plugin
import numpy as np


@register_plugin("tolerance_calculator", "perturbed_dataset_reclustering")
class PerturbedDatasetReclustering(ToleranceCalculator):
    def __init__(self,
        duplicate_detection_algorithm_object: DuplicateDetectionAlgorithm,
        tolerance_dataset_array: np.ndarray = np.array([]),
        perturbations_per_vector: int = 1,
        perturbation_scale: float = 0.1,
        binary_search_steps: int = 20,
        target_structures: int | None = None,):
        super().__init__(
            duplicate_detection_algorithm_object=duplicate_detection_algorithm_object,
            tolerance_dataset_array=tolerance_dataset_array,
            perturbations_per_vector=perturbations_per_vector,
            perturbation_scale=perturbation_scale,
            binary_search_steps=binary_search_steps,
        )
        if target_structures is None:
            target_structures = len(duplicate_detection_algorithm_object.dataset_array)
        self.target_structures = target_structures

    def __str__(self) -> str:
        return f"PerturbedDatasetReclustering(perturbations_per_vector={self.perturbations_per_vector}, perturbation_scale={self.perturbation_scale}, dda={str(self.duplicate_detection_algorithm_object).split('(')[0]})"
    
    def calculate_tolerance(self) -> float:
        """Find the highest and lowest values of tolerance that yield the length of the original dataset from the perturbed dataset, then returns the average of those two values.
        
        Returns:
            float: The calculated tolerance value.
        """
        self._ensure_perturbed_dataset()
        
        low_tolerance = self.binary_search_tolerance(
            target_unique_vectors=self.target_structures,
            find_largest_tolerance_for_target=False,
        )
        high_tolerance = self.binary_search_tolerance(
            target_unique_vectors=self.target_structures,
            find_largest_tolerance_for_target=True,
        )
        return (low_tolerance + high_tolerance) / 2
