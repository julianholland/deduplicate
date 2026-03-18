from deduplicate.core.duplicate_detection_algorithm import DuplicateDetectionAlgorithm
from deduplicate.core.plugin_registry import register_plugin
import numpy as np


@register_plugin(kind="duplicate_detection_algorithm", name="multi_hashing")
class MultiHashing(DuplicateDetectionAlgorithm):
    def __init__(
        self,
        tolerance: float,
        input_vector: np.ndarray,
        dataset_array: np.ndarray,
        perturbations: int,
        acceptance_threshold: int,
        pertrubation_array: np.ndarray,
        distance_metric: str = "euclidean",
        distance_matrix: np.ndarray = np.array([]),
        hash_vector_array: np.ndarray = np.array([]),
    ) -> None:
        super().__init__(
            tolerance, input_vector, dataset_array, distance_metric, distance_matrix
        )
        self.acceptance_threshold = acceptance_threshold
        self.perturbations = perturbations
        self.hash_vector_array = hash_vector_array
        self.perturbation_array = pertrubation_array

    def __str__(self) -> str:
        return f"MultiHashing(tolerance={self.tolerance}, perturbations={self.perturbations}, acceptance_threshold={self.acceptance_threshold})"

    def create_perturbation_array(self, override: bool = False) -> None:
        if self.perturbation_array is not None and not override:
            print(
                "Perturbation array already exists. Skipping creation. To override, set override=True."
            )
        else:
            self.perturbation_array = np.abs(
                np.random.normal(loc=0.0, scale=1.0, size=self.perturbations)
            )

    def round_to_tolerance(self, input_global_descriptor: np.ndarray) -> np.ndarray:
        rounded_vector = (
            np.round(input_global_descriptor / self.tolerance) * self.tolerance
        )
        return rounded_vector

    def create_hash_vector(
        self, input_global_descriptor: np.ndarray, perturbation_array: np.ndarray = None
    ) -> np.ndarray:
        if perturbation_array is None:
            self.create_perturbation_array()
        hash_vector = np.zeros(self.perturbations)
        for i in range(self.perturbations):
            hash_vector[i] = hash(
                input_global_descriptor * perturbation_array[i].tobytes()
            )
        return hash_vector

    def create_hash_vector_array(self, dataset_array: np.ndarray) -> np.ndarray:
        hash_vector_array = np.zeros((dataset_array.shape[0], self.perturbations))
        perturbation_array = self.create_perturbation_array()
        for i in range(dataset_array.shape[0]):
            hash_vector_array[i] = self.create_hash_vector(
                dataset_array[i], perturbation_array
            )

        return hash_vector_array

    def compute_clash_array(self) -> np.ndarray:
        pass
        # perturbation_array = self.create_perturbation_array()
        # clash_array = np.zeros(
        #     (self.perturbations, self.dataset_array.shape[0]), dtype=bool
        # )
        # for i in range(self.dataset_array.shape[0]):
        #     perturbed_array = (
        #         self.round_to_tolerance(self.dataset_array[i]) * perturbation_array
        #     )

    def duplicate_check(self) -> np.bool_:
        """duplicate check against dataset using multi-hashing approach.

        Returns:
            bool: True if a duplicate is detected, False otherwise.
        """
        hash_vector = self.create_hash_vector(self.input_vector)

        clash_vector = np.zeros(self.perturbations, dtype=bool)
        for hash_index, hash_value in enumerate(hash_vector):
            if hash_value in self.hash_vector_array[hash_index]:
                clash_vector[hash_index] = True
        duplicate_vote_array = np.sum(clash_vector) / self.perturbations

        duplicate_structure = duplicate_vote_array >= self.acceptance_threshold

        return bool(duplicate_structure)

    def get_dataset_unique_structures(self) -> int:
        pass
