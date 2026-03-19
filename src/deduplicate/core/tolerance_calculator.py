import numpy as np
from abc import ABC, abstractmethod
from deduplicate.core.duplicate_detection_algorithm import DuplicateDetectionAlgorithm


class ToleranceCalculator(ABC):
    def __init__(
        self,
        dataset_array: np.ndarray,
        duplicate_detection_algorithm_object: DuplicateDetectionAlgorithm,
        perturbations_per_vector: int = 1,
        perturbation_scale: float = 0.1,
    ) -> None:
        self.dataset_array = dataset_array
        self.duplicate_detection_algorithm_object = duplicate_detection_algorithm_object
        self.perturbations_per_vector = perturbations_per_vector
        self.perturbation_scale = perturbation_scale
        
    def binary_search_tolerance(
        self, target_structures: int, find_largest_tolerance_for_target: bool = True
    ):
        best_min = 0.0
        best_max = np.max(self.dataset_array) - np.min(self.dataset_array)
        tolerance = best_max / 2

        exact_list = []
        all_diff_dict = {}

        for _ in range(self.binary_search_steps):
            self.duplicate_detection_algorithm_object.tolerance = tolerance
            unique_structures = self.duplicate_detection_algorithm_object.get_dataset_unique_structures()
            all_diff_dict[tolerance] = np.abs(unique_structures - target_structures)

            if unique_structures < target_structures:
                best_max = tolerance
            elif unique_structures > target_structures:
                best_min = tolerance
            else:
                exact_list.append(tolerance)
                if (target_structures == 1) or not find_largest_tolerance_for_target:
                    best_max = tolerance
                elif (
                    target_structures
                    == len(self.duplicate_detection_algorithm_object.dataset_array)
                    or find_largest_tolerance_for_target
                ):
                    best_min = tolerance
                else:
                    best_max = tolerance

            tolerance = (best_min + best_max) / 2

        if len(exact_list) > 0:
            if find_largest_tolerance_for_target:
                tolerance = max(exact_list)
            else:
                tolerance = min(exact_list)
        else:
            tolerance = [
                tol
                for tol, diff in all_diff_dict.items()
                if diff == min(all_diff_dict.values())
            ][0]

        return tolerance

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

    @abstractmethod
    def calculate_tolerance(self) -> float:
        pass
