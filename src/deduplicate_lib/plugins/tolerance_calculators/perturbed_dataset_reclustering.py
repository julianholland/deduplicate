from deduplicate_lib.core.tolerance_calculator import ToleranceCalculator
from deduplicate_lib.core.plugin_registry import register_plugin


@register_plugin("tolerance_calculator", "perturbed_dataset_reclustering")
class PerturbedDatasetReclustering(ToleranceCalculator):
    def __str__(self) -> str:
        return f"PerturbedDatasetReclustering(perturbations_per_vector={self.perturbations_per_vector}, perturbation_scale={self.perturbation_scale}, dda={str(self.duplicate_detection_algorithm_object).split('(')[0]})"
    
    def calculate_tolerance(self) -> float:
        """Find the highest and lowest values of tolerance that yield the length of the original dataset from the perturbed dataset, then returns the average of those two values.
        
        Returns:
            float: The calculated tolerance value.
        """
        self._ensure_perturbed_dataset()
        
        low_tolerance = self.binary_search_tolerance(
            target_unique_vectors=len(self.duplicate_detection_algorithm_object.dataset_array),
            find_largest_tolerance_for_target=False,
        )
        high_tolerance = self.binary_search_tolerance(
            target_unique_vectors=len(self.duplicate_detection_algorithm_object.dataset_array),
            find_largest_tolerance_for_target=True,
        )
        return (low_tolerance + high_tolerance) / 2
