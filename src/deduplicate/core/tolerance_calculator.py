import numpy as np
from abc import ABC, abstractmethod
from deduplicate.core.duplicate_detection_algorithm import DuplicateDetectionAlgorithm
from contextlib import contextmanager
import warnings


class ToleranceCalculator(ABC):
    def __init__(
        self,
        duplicate_detection_algorithm_object: DuplicateDetectionAlgorithm,
        tolerance_dataset_array: np.ndarray = np.array([]),
        perturbations_per_vector: int = 1,
        perturbation_scale: float = 0.1,
        binary_search_steps: int = 20,
    ) -> None:
        self.tolerance_dataset_array = tolerance_dataset_array
        self.duplicate_detection_algorithm_object = duplicate_detection_algorithm_object
        self.perturbations_per_vector = perturbations_per_vector
        self.perturbation_scale = perturbation_scale
        self.binary_search_steps = binary_search_steps

    @contextmanager
    def temp_attr(self, obj, attr, value):
        original = getattr(obj, attr)
        setattr(obj, attr, value)
        try:
            yield
        finally:
            setattr(obj, attr, original)

    def binary_search_tolerance(
        self,
        target_unique_vectors: int,
        find_largest_tolerance_for_target: bool = True,
    ) -> float:
        """
        Binary search to find the tolerance that yields the target number of unique structures.
        """
        best_min = 0.0
        # to ensure that our original best max value is above the tolerance value that yields all structures as the same, we tie it to the relative standard deviation of the dataset array. A more diverse dataset will have a higher standard deviation and thus require a higher tolerance to yield all structures as the same, while a more homogenous dataset will have a lower standard deviation and thus require a lower tolerance to yield all structures as the same.
        ptp=np.ptp(self.tolerance_dataset_array) 
        abs_std=np.mean(np.std(self.tolerance_dataset_array, axis=0))
        best_max = (ptp  + abs_std) * self.duplicate_detection_algorithm_object.dataset_array.shape[1]      
        tolerance = best_max 

        exact_list = []
        all_diff_dict = {}

        if self.binary_search_steps <= 0:
            warnings.warn(f"Binary search called with {self.binary_search_steps} steps. No binary search performed.\n Returning tolerance of half the range of the perturbed dataset: {tolerance}.")
            return tolerance
        
        # temporarily set the dataset_array to the tolerance_dataset_array for the duration of the search
        print(self.duplicate_detection_algorithm_object.distance_matrix.shape)
        with self.temp_attr(
            self.duplicate_detection_algorithm_object,
            "dataset_array",
            self.tolerance_dataset_array,
        ):
            for _ in range(self.binary_search_steps):
                # temporarily set the tolerance for the duration of this iteration of the search
                with self.temp_attr(
                    self.duplicate_detection_algorithm_object, "tolerance", tolerance
                ):
                    
                    unique_vectors = self.duplicate_detection_algorithm_object.get_dataset_unique_structures()
                all_diff_dict[tolerance] = unique_vectors
                if unique_vectors < target_unique_vectors: # loosen tolerance
                    best_max = tolerance
                elif unique_vectors > target_unique_vectors: # tighten tolerance
                    best_min = tolerance

                else:
                    exact_list.append(tolerance)
                    if (
                        target_unique_vectors
                        == len(
                            self.duplicate_detection_algorithm_object.dataset_array
                        )
                        or find_largest_tolerance_for_target
                    ):
                        best_min = tolerance
                    else:
                        best_max = tolerance

                tolerance = (best_min + best_max) / 2
        # Find the best exact value if it exists, otherwise find the closest value
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
            warnings.warn(f"No exact tolerance found for target of {target_unique_vectors} unique vectors during binary search.\n Returning closest tolerance found: {tolerance} with {all_diff_dict[tolerance]} unique vectors.")
        print(all_diff_dict)
        return tolerance

    def create_perturbed_dataset(self, seed: int = 803):
        """
        Create a perturbed dataset by adding random noise to the original dataset. The original dataset is included as the first vector for each original vector, followed by the perturbed versions of that vector.
        """        
        rng=np.random.default_rng(seed)
        self.tolerance_dataset_array = np.zeros(
            (
                len(self.duplicate_detection_algorithm_object.dataset_array)
                * self.perturbations_per_vector,
                self.duplicate_detection_algorithm_object.dataset_array.shape[1],
            )
        )

        for i, vector in enumerate(self.duplicate_detection_algorithm_object.dataset_array):
            self.tolerance_dataset_array[i * self.perturbations_per_vector] = (
                self.duplicate_detection_algorithm_object.dataset_array[i]
            )
            for j in range(1, self.perturbations_per_vector):
                perturbation = rng.normal(
                    loc=0.0, scale=self.perturbation_scale, size=vector.shape
                )
                perturbed_vector = vector + perturbation
                self.tolerance_dataset_array[i * self.perturbations_per_vector + j] = (
                    perturbed_vector
                )

    def _ensure_perturbed_dataset(self):
        if (
            self.tolerance_dataset_array.shape[0]
            != len(self.duplicate_detection_algorithm_object.dataset_array)
            * self.perturbations_per_vector
        ):
            warnings.warn(
                "Perturbed dataset is not properly initialized. Recreating it.\nEnsure a perturbed dataset is set prior finding tolerance."
            )
            self.create_perturbed_dataset()

    @abstractmethod
    def calculate_tolerance(self) -> float:
        pass
