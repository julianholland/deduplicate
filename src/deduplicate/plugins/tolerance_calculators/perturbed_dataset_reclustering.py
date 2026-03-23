from deduplicate.core.tolerance_calculator import ToleranceCalculator
from deduplicate.core.duplicate_detection_algorithm import DuplicateDetectionAlgorithm
from deduplicate.core.plugin_registry import register_plugin
import numpy as np
import warnings


@register_plugin("tolerance_calculator", "perturbed_dataset_reclustering")
class PerturbedDatasetReclustering(ToleranceCalculator):
    def calculate_tolerance(self) -> float:
        """Find the highest and lowest values of tolerance that yield the length of the original dataset from the perturbed dataset, then returns the average of those two values.
        
        Returns:
            float: The calculated tolerance value.
        """
        self._ensure_perturbed_dataset()
        self.duplicate_detection_algorithm_object.dataset_array = self.perturbed_dataset
        low_tolerance = self.binary_search_tolerance(
            target_structures=len(self.dataset_array),
            find_largest_tolerance_for_target=False,
        )
        high_tolerance = self.binary_search_tolerance(
            target_structures=len(self.dataset_array),
            find_largest_tolerance_for_target=True,
        )
        return (low_tolerance + high_tolerance) / 2
