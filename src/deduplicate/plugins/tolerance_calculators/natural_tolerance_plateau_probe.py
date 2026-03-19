from ase import data

from deduplicate.core.tolerance_calculator import ToleranceCalculator
from deduplicate.core.duplicate_detection_algorithm import DuplicateDetectionAlgorithm
from deduplicate.core.plugin_registry import register_plugin
import numpy as np
import warnings


@register_plugin("tolerance_calculator", "natural_tolerance_plateau_probe")
class NaturalTolerancePlateauProbe(ToleranceCalculator):
    def __init__(
        self, 
        dataset_array: np.ndarray, 
        duplicate_detection_algorithm_object: DuplicateDetectionAlgorithm,
        perturbed_dataset: np.ndarray = np.array([]),
        binary_search_steps: int = 50,
    ) -> None:
        super().__init__(
            dataset_array=dataset_array,
            duplicate_detection_algorithm_object=duplicate_detection_algorithm_object,
            perturbed_dataset=perturbed_dataset,
            
        )
        self.perturbed_dataset = perturbed_dataset
        self.binary_search_steps = binary_search_steps

    def _ensure_perturbed_dataset(self):
        if self.perturbed_dataset.shape[0] != len(self.dataset_array) * self.perturbations_per_vector:
            warnings.warn("Perturbed dataset is not properly initialized. Recreating it.\nEnsure a perturbed dataset is set prior finding tolerance.")
            self.create_perturbed_dataset()
    

    def tolerance_probe(self, lower_tolerance: float, upper_tolerance: float, tolerance_steps: float) -> dict:
        tolerance_results = {}
        
        for tol in np.linspace(lower_tolerance, upper_tolerance, tolerance_steps):
            self.duplicate_detection_algorithm_object.tolerance = tol
            unique_structures = self.duplicate_detection_algorithm_object.get_dataset_unique_structures()
            tolerance_results[tol] = unique_structures
        return tolerance_results


    def find_plateaus(self, tolerance_results: dict) -> tuple: # check this ai slop
        dataset_size = len(self.dataset_array)
        datapoints_to_calculate_gradient = min(5, dataset_size - 1)
        sorted_tols = sorted(tolerance_results.keys())
        unique_counts = [tolerance_results[tol] for tol in sorted_tols]

        plateau_log = np.zeros(len(sorted_tols), dtype=bool)
        for i in range(1, len(unique_counts)-datapoints_to_calculate_gradient):
            gradient = (unique_counts[i + datapoints_to_calculate_gradient] - unique_counts[i]) / (sorted_tols[i + datapoints_to_calculate_gradient] - sorted_tols[i])
            if abs(gradient) < 1e-3:  # Threshold for plateau detection
                plateau_log[i] = True

        plateau_lengths=[]
        for i in range(len(plateau_log)):
            if plateau_log[i]:
                if not plateau_log[i-1]:
                    start = i
            else:
                if plateau_log[i-1]:
                    end = i
                    plateau_lengths.append((sorted_tols[start], sorted_tols[end], end - start))

        return plateau_lengths
        
    def calculate_tolerance(self) -> float:
        self._ensure_perturbed_dataset()
        all_same_tolerance = self.binary_search_tolerance(target_structures=1, find_largest_tolerance_for_target=False)
        all_different_tolerance = self.binary_search_tolerance(target_structures=len(self.dataset_array), find_largest_tolerance_for_target=True)
        probe_results = self.tolerance_probe(all_same_tolerance, all_different_tolerance, self.binary_search_steps)
        plateaus = self.find_plateaus(probe_results)
        if len(plateaus) == 0:
            warnings.warn("No plateaus found in tolerance probe. Consider adding in perturbed structures and/or increasing dataset size.\nReturning average of all same and all different tolerance as fallback.")
            return (all_same_tolerance + all_different_tolerance) / 2
        else:
            longest_plateau = max(plateaus, key=lambda x: x[2])
            return (longest_plateau[0] + longest_plateau[1]) / 2