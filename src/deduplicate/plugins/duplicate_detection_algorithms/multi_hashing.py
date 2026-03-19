from deduplicate.core.duplicate_detection_algorithm import (
    DuplicateDetectionAlgorithm,
    hamming_distance,
)
from deduplicate.core.plugin_registry import register_plugin

import numpy as np
import warnings


@register_plugin(kind="duplicate_detection_algorithm", name="multi_hashing")
class MultiHashing(DuplicateDetectionAlgorithm):
    ALLOWED_DISTANCES = {"hamming": hamming_distance}

    def __init__(
        self,
        tolerance: float,
        input_vector: np.ndarray,
        dataset_array: np.ndarray,
        perturbations: int = 200,
        seed: int = 803,
        pertrubation_array: np.ndarray = np.array([]),
        acceptance_threshold: float = 0.5,
        distance_metric: str = "hamming",
        distance_matrix: np.ndarray = np.array([]),
        hash_vector_array: np.ndarray = np.array([]),
    ) -> None:
        super().__init__(
            tolerance=tolerance,
            input_vector=input_vector,
            dataset_array=dataset_array,
            distance_metric=distance_metric,
            distance_matrix=distance_matrix,
        )
        self.seed = seed
        self.acceptance_threshold = acceptance_threshold
        self.perturbations = perturbations
        self.hash_vector_array = hash_vector_array
        self.perturbation_array = pertrubation_array

    def __str__(self) -> str:
        return f"MultiHashing(tolerance={self.tolerance}, perturbations={self.perturbations}, acceptance_threshold={self.acceptance_threshold})"

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

    def round_to_tolerance(self) -> np.ndarray:
        rounded_vector = np.round(self.input_vector / self.tolerance) * self.tolerance
        return rounded_vector

    def create_hash_vector(self) -> np.ndarray:
        self._ensure_perturbation_array()
        hash_vector = np.zeros(self.perturbations)
        for i in range(self.perturbations):
            hash_vector[i] = hash(
                (self.round_to_tolerance() * self.perturbation_array[i]).tobytes()
            )
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
            self.input_vector = self.dataset_array[i]
            hash_vector = self.create_hash_vector()
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

        duplicate_structure = duplicate_vote_array >= self.acceptance_threshold

        return bool(duplicate_structure)

    def get_dataset_unique_structures(self) -> int:
        """Counts the number of unique structures in the dataset by checking for hash value clashes across the dataset for each perturbation.

        Returns:
            int: The number of unique structures in the dataset.
        """
        clash_array = np.zeros((self.hash_vector_array.shape), dtype=bool)
        clash_array[np.nonzero(self.hash_vector_array)] = True
        
        return np.sum(
            np.sum(clash_array, axis=1) / self.perturbations
            >= self.acceptance_threshold
        )
