
from deduplicate.core.tolerance_calculator import ToleranceCalculator
from deduplicate.core.duplicate_detection_algorithm import DuplicateDetectionAlgorithm
from deduplicate.core.plugin_registry import register_plugin
import numpy as np
import warnings


@register_plugin("tolerance_calculator", "natural_tolerance_plateau_probe")
class NaturalTolerancePlateauProbe(ToleranceCalculator):
    def __init__(
        self,
        duplicate_detection_algorithm_object: DuplicateDetectionAlgorithm,
        tolerance_dataset_array: np.ndarray = np.array([]),
        perturbations_per_vector: int = 1,
        perturbation_scale: float = 0.1,
        binary_search_steps: int = 50,
        probe_steps: int = 100,
    ) -> None:
        super().__init__(
            duplicate_detection_algorithm_object=duplicate_detection_algorithm_object,
            tolerance_dataset_array=tolerance_dataset_array,
            perturbations_per_vector=perturbations_per_vector,
            perturbation_scale=perturbation_scale,
            binary_search_steps=binary_search_steps,
        )
        self.probe_steps = probe_steps

    def __str__(self) -> str:
        return f"NaturalTolerancePlateauProbe(perturbations_per_vector={self.perturbations_per_vector}, perturbation_scale={self.perturbation_scale}, dda={str(self.duplicate_detection_algorithm_object).split('(')[0]})"

    def tolerance_probe(
        self, lower_tolerance: float, upper_tolerance: float, tolerance_steps: float
    ) -> dict:
        tolerance_results = {}
        with self.temp_attr(
            self.duplicate_detection_algorithm_object,
            "dataset_array",
            self.tolerance_dataset_array,
        ):
            for tol in np.linspace(lower_tolerance, upper_tolerance, tolerance_steps):
                with self.temp_attr(
                    self.duplicate_detection_algorithm_object, "tolerance", tol
                ):
                    self.duplicate_detection_algorithm_object.tolerance = tol
                    unique_structures = self.duplicate_detection_algorithm_object.get_dataset_unique_structures()
                    tolerance_results[tol] = unique_structures
            return tolerance_results

    def find_plateaus(self, tolerance_results: dict, datapoints_to_calculate_gradient: int = 3, plateau_threshold: float = 1e-3) -> tuple:  
        sorted_tols = sorted(tolerance_results.keys())
        unique_counts = [tolerance_results[tol] for tol in sorted_tols]

        # detect plateaus by calculating the gradient of unique_counts with respect to tolerance and finding where it is close to zero
        plateau_log = np.zeros(len(sorted_tols), dtype=bool)
        for i in range(len(unique_counts) - datapoints_to_calculate_gradient):
            gradient = (
                unique_counts[i + datapoints_to_calculate_gradient] - unique_counts[i]
            ) / (sorted_tols[i + datapoints_to_calculate_gradient] - sorted_tols[i])
            if abs(gradient) < plateau_threshold:  # Threshold for plateau detection
                plateau_log[i] = True
        # Record the start and end of each plateau and its length
        plateau_lengths = []
        for i in range(len(plateau_log)):
            if plateau_log[i]:
                if not plateau_log[i - 1]:
                    start = i
            else:
                if plateau_log[i - 1]:
                    end = i
                    plateau_lengths.append(
                        (sorted_tols[start], sorted_tols[end], end - start)
                    )

        return plateau_lengths

    def calculate_tolerance(self) -> float:
        self._ensure_perturbed_dataset()
        all_same_tolerance = self.binary_search_tolerance(
            target_unique_vectors=1, find_largest_tolerance_for_target=False
        )
        all_different_tolerance = self.binary_search_tolerance(
            target_unique_vectors=len(self.tolerance_dataset_array),
            find_largest_tolerance_for_target=True,
        )
        probe_results = self.tolerance_probe(
            all_same_tolerance, all_different_tolerance, self.probe_steps
        )
        plateaus = self.find_plateaus(probe_results)
        if len(plateaus) == 0:
            warnings.warn(
                "No plateaus found in tolerance probe. Consider adding in perturbed structures and/or increasing dataset size.\nReturning average of all same and all different tolerance as fallback."
            )
            return (all_same_tolerance + all_different_tolerance) / 2
        else:
            longest_plateau = max(plateaus, key=lambda x: x[2])
            return (longest_plateau[0] + longest_plateau[1]) / 2
