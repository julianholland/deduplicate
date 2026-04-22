from deduplicate_lib.core.tolerance_calculator import ToleranceCalculator
from deduplicate_lib.core.duplicate_detection_algorithm import DuplicateDetectionAlgorithm
from deduplicate_lib.core.plugin_registry import register_plugin
import numpy as np


@register_plugin("tolerance_calculator", "perturbed_dataset_reclustering")
class PerturbedDatasetReclustering(ToleranceCalculator):
    """A tolerance calculator that creates a perturbed dataset from the original dataset, then uses the duplicate detection algorithm to find the tolerance that yields the same number of unique structures in the perturbed dataset as in the target. The tolerance can be calculated as 'loose' (the highest tolerance that yields the target number of unique structures), 'tight' (the lowest tolerance that yields the target number of unique structures), or 'average' (the average of the highest and lowest tolerances that yield the target number of unique structures).

    Args:
        duplicate_detection_algorithm_object (DuplicateDetectionAlgorithm): The duplicate detection algorithm object to use for calculating the tolerance.
        tolerance_dataset_array (np.ndarray, optional): The dataset array to use for calculating the tolerance. If not provided, the dataset array from the duplicate detection algorithm object will be used. Defaults to np.array([]).
        perturbations_per_vector (int, optional): The number of perturbations to create for each vector in the dataset array. Defaults to 1.
        perturbation_scale (float, optional): The scale of the perturbations to create. Defaults to 0.1.
        binary_search_steps (int, optional): The number of steps to use for the binary search to find the tolerance. Defaults to 20.
        target_structures (int | None, optional): The target number of unique structures to find the tolerance for. If None, the target will be the number of unique structures in the original dataset array. Defaults to None.
        target_structures_threshold (str, optional): The method to use for calculating the tolerance threshold. Can be 'average', 'loose', or 'tight'. Defaults to 'average'.

    Functions:
        calculate_tolerance: Calculate the tolerance value based on the perturbed dataset and the target number of unique structures.

        Returns:
            float: The calculated tolerance value.
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
        self._ensure_perturbed_dataset()
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