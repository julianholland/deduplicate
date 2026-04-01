from deduplicate_lib.core.tolerance_calculator import ToleranceCalculator
from deduplicate_lib.core.duplicate_detection_algorithm import (
    DuplicateDetectionAlgorithm,
)
from deduplicate_lib.core.plugin_registry import register_plugin
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
        probe_buffer_fraction: float = 0.1,
    ) -> None:
        super().__init__(
            duplicate_detection_algorithm_object=duplicate_detection_algorithm_object,
            tolerance_dataset_array=tolerance_dataset_array,
            perturbations_per_vector=perturbations_per_vector,
            perturbation_scale=perturbation_scale,
            binary_search_steps=binary_search_steps,
        )
        self.probe_steps = probe_steps
        self.probe_buffer_fraction = probe_buffer_fraction

    def __str__(self) -> str:
        return f"NaturalTolerancePlateauProbe(perturbations_per_vector={self.perturbations_per_vector}, perturbation_scale={self.perturbation_scale}, dda={str(self.duplicate_detection_algorithm_object).split('(')[0]})"

    def tolerance_probe(
        self, lower_tolerance: float, upper_tolerance: float, tolerance_steps: int
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

    def get_plateau_log(
        self,
        sorted_tols: list,
        tolerance_results: dict,
        datapoints_to_calculate_gradient: int = 3,
        plateau_threshold: float = 1e-3,
        # plot_plateaus: bool = False,
    ) -> np.ndarray:
        
        if datapoints_to_calculate_gradient <= 1:
            raise ValueError("datapoints_to_calculate_gradient must be greater than 1.")
        
        if len(sorted_tols) < datapoints_to_calculate_gradient:
            raise ValueError("Not enough tolerance steps to calculate gradient with the given datapoints_to_calculate_gradient.")

        unique_counts = [tolerance_results[tol] for tol in sorted_tols]
        total_gradient = (unique_counts[-1] - unique_counts[0]) / (sorted_tols[-1] - sorted_tols[0])
        relative_plateau_threshold = plateau_threshold * total_gradient
        
        # detect plateaus by calculating the gradient of unique_counts with respect to tolerance and finding where it is close to zero
        plateau_log = np.zeros(len(sorted_tols) - datapoints_to_calculate_gradient, dtype=bool)
        gradient_log = np.zeros(len(sorted_tols) - datapoints_to_calculate_gradient)
        for i in range(len(unique_counts) - datapoints_to_calculate_gradient):
            gradient = (
                unique_counts[i + datapoints_to_calculate_gradient] - unique_counts[i]
            ) / (sorted_tols[i + datapoints_to_calculate_gradient] - sorted_tols[i])
            gradient_log[i] = gradient
            if (
                abs(gradient) < abs(relative_plateau_threshold)
            ):  # Threshold for plateau detection
                plateau_log[i] = True

        
        # if plot_plateaus:
        #     plt.clf()
        #     x_vals = sorted_tols[:-datapoints_to_calculate_gradient]
        #     y_vals = unique_counts[:-datapoints_to_calculate_gradient]
        #     fig, ax = plt.subplots()
        #     # Plot line for context
        #     ax.plot(x_vals, y_vals, color='gray', alpha=0.5, label="Unique structures")
        #     # Scatter plot colored by gradient
        #     sc = ax.scatter(x_vals, y_vals, c=gradient_log, cmap='coolwarm', label="Gradient")
        #     ax.set_xlabel("Tolerance")
        #     ax.set_ylabel("unique structures")
        #     ax.set_title("Plateau Detection")
        #     fig.colorbar(sc, ax=ax, label='Gradient')
        #     ax.legend()
        #     fig.savefig(f'plateau_detection_{str(self.duplicate_detection_algorithm_object).split("(")[0]}.png')

        # Plot line for context

        return plateau_log

    def find_plateaus(
        self,
        tolerance_results: dict,
        datapoints_to_calculate_gradient: int = 3,
        plateau_threshold: float = 1e-3,
    ) -> tuple:
        sorted_tols = sorted(tolerance_results.keys())
        plateau_log = self.get_plateau_log(
            sorted_tols,
            tolerance_results,
            datapoints_to_calculate_gradient,
            plateau_threshold,
        )

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
            target_unique_vectors=int(
                np.max([1, np.floor(self.probe_buffer_fraction * len(self.tolerance_dataset_array))])
            ),
            find_largest_tolerance_for_target=False,
        )
        all_different_tolerance = self.binary_search_tolerance(
            target_unique_vectors=len(self.tolerance_dataset_array) - int(
                np.floor(self.probe_buffer_fraction * len(self.tolerance_dataset_array))
            ),
            find_largest_tolerance_for_target=True,
        )
        probe_results = self.tolerance_probe(
            all_same_tolerance, all_different_tolerance, self.probe_steps
        )
        plateaus = self.find_plateaus(probe_results)
        if len(plateaus) == 0:
            warnings.warn(
                "No plateaus found in tolerance probe. Consider adding in perturbed structures and/or increasing dataset size and/or increaseing probe steps.\nReturning average of all same and all different tolerance as fallback."
            )
            return (all_same_tolerance + all_different_tolerance) / 2
        else:
            longest_plateau = max(plateaus, key=lambda x: x[2])
            return (longest_plateau[0] + longest_plateau[1]) / 2
