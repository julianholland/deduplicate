from deduplicate_lib.core.duplicate_detection_algorithm import (
    DuplicateDetectionAlgorithm,
    hamming_distance,
)
from deduplicate_lib.core.plugin_registry import register_plugin

import numpy as np
import warnings
from numba import njit

@njit
def fast_round_and_perturb(input_vector, perturbation_array, tolerance):
    n_pert = perturbation_array.shape[0]
    n_feat = input_vector.shape[0]
    out = np.empty((n_pert, n_feat), dtype=input_vector.dtype)

    for i in range(n_pert):
        p = perturbation_array[i]
        for j in range(n_feat):
            v = p * input_vector[j]
            out[i, j] = np.round(v / tolerance) * tolerance

    return out 
# def fast_round_and_perturb(input_vector, perturbation_array, tolerance):
#     perturbed_vector_array = np.array([perturbation_array[i] * input_vector for i in range(perturbation_array.shape[0])])
#     round_and_perturb_array = np.round(perturbed_vector_array / tolerance) * tolerance # pragma: no cover, tested but does not appear in coverage report due to numba jit compilation
    
#     return round_and_perturb_array # pragma: no cover

@register_plugin(kind="duplicate_detection_algorithm", name="multi_hashing")
class MultiHashing(DuplicateDetectionAlgorithm):
    ALLOWED_DISTANCES = {"hamming": hamming_distance}

    def __init__(
        self,
        tolerance: float = 0.1,
        input_vector: np.ndarray = np.array([]),
        dataset_array: np.ndarray = np.array([]),
        perturbations: int = 200,
        seed: int = 803,
        pertrubation_array: np.ndarray = np.array([]),
        distance_metric: str = "hamming",
        sigma_accepatnce_threshold: int = 1,
        distance_matrix: np.ndarray = np.array([]),
        hash_vector_array: np.ndarray = np.array([]),
        unique_vector_indices: np.ndarray = np.array([]),
    ) -> None:
        super().__init__(
            tolerance=tolerance,
            input_vector=input_vector,
            dataset_array=dataset_array,
            distance_metric=distance_metric,
            distance_matrix=distance_matrix,
            unique_vector_indices=unique_vector_indices,
        )
        self.seed = seed
        self.sigma_accepatnce_threshold = sigma_accepatnce_threshold
        self.perturbations = perturbations
        self.hash_vector_array = hash_vector_array
        self.perturbation_array = pertrubation_array

    def __str__(self) -> str:
        return f"MultiHashing(tolerance={self.tolerance}, perturbations={self.perturbations}, sigma_acceptance_threshold={self.sigma_accepatnce_threshold})"

    def set_perturbation_array(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.perturbation_array = np.abs(
            rng.normal(loc=0.0, scale=1.0, size=self.perturbations)
        )

    def _ensure_perturbation_array(self):
        
        if self.perturbation_array.shape[0] != self.perturbations:
            warnings.warn(
                "Perturbation array shape does not match expected number of perturbations; recomputing. \nAssign the perturbation array prior duplication checks to avoid this warning."
            )
            self.set_perturbation_array()

    def get_acceptance_threshold(self) -> float:
        if self.sigma_accepatnce_threshold < 1 or self.sigma_accepatnce_threshold > 4 or not isinstance(self.sigma_accepatnce_threshold, int):
            raise ValueError("Sigma acceptance threshold must be an integer between 1 and 4 inclusive.")
        sigma_dict = {
            1: 0.682689492137086,
            2: 0.954499736103642,
            3: 0.997300203936740,
            4: 0.999936657516333,
        }
        return sigma_dict[self.sigma_accepatnce_threshold]

    def round_to_tolerance(self) -> np.ndarray:
        rounded_vector = np.round(self.input_vector / self.tolerance) * self.tolerance
        return rounded_vector

    def create_hash_vector(self, input_vector: np.ndarray | None = None) -> np.ndarray:
        if input_vector is None:
            input_vector = self.input_vector
        self._ensure_perturbation_array()
        rounded_and_perturbed = fast_round_and_perturb(input_vector, self.perturbation_array, self.tolerance)
        hash_vector = np.zeros(self.perturbations, dtype=int)
        for i in range(self.perturbations):
            hash_vector[i] = hash(rounded_and_perturbed[i].tobytes())
        return hash_vector

    def create_hash_vector_array(self) -> np.ndarray:
        """Creates the hash vector array for the dataset. Only unique hash values are stored for each perturbation.

        Returns:
            np.ndarray: A 2D array containing the hash values for each vector in the dataset for each perturbation.
        """
        self.hash_vector_array = np.zeros(
            (self.dataset_array.shape[0], self.perturbations), dtype=int
        )
        self._ensure_perturbation_array()
        for i in range(self.dataset_array.shape[0]):
            hash_vector = self.create_hash_vector(self.dataset_array[i])
            for j in range(self.perturbations):
                if hash_vector[j] not in self.hash_vector_array[:, j]:
                    self.hash_vector_array[i, j] = hash_vector[j]
        return self.hash_vector_array

    def _ensure_hash_vector_array(self):
        if self.hash_vector_array.shape[0] != self.dataset_array.shape[0]:
            warnings.warn(
                "Hash vector array shape does not match dataset; recomputing."
            )
            self.create_hash_vector_array()

    def duplicate_check(self) -> bool:
        """duplicate check against dataset using multi-hashing approach.

        Returns:
            bool: True if a duplicate is detected, False otherwise.
        """
        self._ensure_hash_vector_array()
        hash_vector = self.create_hash_vector()

        clash_vector = np.zeros(self.perturbations, dtype=bool)
        for hash_index, hash_value in enumerate(hash_vector):
            if hash_value in self.hash_vector_array[:, hash_index]:
                clash_vector[hash_index] = True
        duplicate_vote_array = np.sum(clash_vector) / self.perturbations

        duplicate_structure = duplicate_vote_array >= self.get_acceptance_threshold()

        return bool(duplicate_structure)
    
    def get_uniqueness_score(self) -> float:
        self._ensure_hash_vector_array()
        hash_vector = self.create_hash_vector()
        clash_vector = np.zeros(self.perturbations, dtype=bool)
        for hash_index, hash_value in enumerate(hash_vector):
            if hash_value in self.hash_vector_array[:, hash_index]:
                clash_vector[hash_index] = True
        uniqueness_score = 1 - np.sum(clash_vector) / self.perturbations

        return uniqueness_score

    def get_dataset_unique_structures(self) -> int:
        """Counts the number of unique structures in the dataset by checking for hash value clashes across the dataset for each perturbation.

        Returns:
            int: The number of unique structures in the dataset.
        """
        self.create_hash_vector_array()
        clash_array = np.zeros((self.hash_vector_array.shape), dtype=bool)
        clash_array[np.nonzero(self.hash_vector_array)] = True
        self.unique_vector_indices = np.sum(clash_array, axis=1) / self.perturbations >= self.get_acceptance_threshold()
        
        return np.sum(self.unique_vector_indices)
    
    def pre_dda_processing(self, *args, **kwargs) -> None:
        self.set_perturbation_array()

    def add_input_vector_to_dda(self) -> None:
        """Add the input vector to the dataset array and update the hash vector array accordingly."""
        self.dataset_array = np.vstack((self.dataset_array, self.input_vector))
        new_hash_vector = self.create_hash_vector()
        self.hash_vector_array = np.vstack((self.hash_vector_array, new_hash_vector))