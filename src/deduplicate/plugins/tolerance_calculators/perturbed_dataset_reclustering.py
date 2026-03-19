from deduplicate.core.tolerance_calculator import ToleranceCalculator
from deduplicate.core.duplicate_detection_algorithm import DuplicateDetectionAlgorithm
from deduplicate.core.plugin_registry import register_plugin
import numpy as np
import warnings


@register_plugin("tolerance_calculator", "perturbed_dataset_reclustering")
class PerturbedDatasetReclustering(ToleranceCalculator):
    def __init__(
        self,
        dataset_array: np.ndarray,
        duplicate_detection_algorithm_object: DuplicateDetectionAlgorithm,
        perturbed_dataset: np.ndarray = np.array([]),
        binary_search_steps: int = 50,
        perturbations_per_vector: int = 5,
        perturbation_scale: float = 0.1,
    ) -> None:
        super().__init__(
            dataset_array=dataset_array,
            duplicate_detection_algorithm_object=duplicate_detection_algorithm_object,
        )
        self.perturbed_dataset = perturbed_dataset
        self.binary_search_steps = binary_search_steps
        self.perturbations_per_vector = perturbations_per_vector
        self.perturbation_scale = perturbation_scale

    def create_perturbed_dataset(self) -> np.ndarray:
        self.perturbed_dataset = np.zeros(
            len(self.dataset_array) * self.perturbations_per_vector,
            self.dataset_array.shape[1],
        )
        for i, vector in enumerate(self.dataset_array):
            for j in range(self.perturbations_per_vector):
                perturbation = np.random.normal(
                    loc=0.0, scale=self.perturbation_scale, size=vector.shape
                )
                perturbed_vector = vector + perturbation
                self.perturbed_dataset[i * self.perturbations_per_vector + j] = (
                    perturbed_vector
                )

    def _ensure_perturbed_dataset(self):
        if (
            self.perturbed_dataset.shape[0]
            != len(self.dataset_array) * self.perturbations_per_vector
        ):
            warnings.warn(
                "Perturbed dataset is not properly initialized. Recreating it.\nEnsure a perturbed dataset is set prior finding tolerance."
            )
            self.create_perturbed_dataset()

    def calculate_tolerance(self) -> float:
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
