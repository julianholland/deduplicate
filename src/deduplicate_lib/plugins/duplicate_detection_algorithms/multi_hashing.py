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
    n_pert = perturbation_array.shape[0] # pragma: no cover
    n_feat = input_vector.shape[0] # pragma: no cover
    out = np.empty((n_pert, n_feat), dtype=input_vector.dtype) # pragma: no cover

    for i in range(n_pert): # pragma: no cover
        p = perturbation_array[i] # pragma: no cover
        for j in range(n_feat): # pragma: no cover
            v = p * input_vector[j]# pragma: no cover
            out[i, j] = np.round(v / tolerance) * tolerance # pragma: no cover

    return out # pragma: no cover

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
        unique_vector_indices: np.ndarray = np.array([]),
        max_vector_array_size: int = 10000,
    ) -> None:
        super().__init__(
            tolerance=tolerance,
            input_vector=input_vector,
            dataset_array=dataset_array,
            distance_metric=distance_metric,
            distance_matrix=distance_matrix,
            unique_vector_indices=unique_vector_indices,
            max_vector_array_size=max_vector_array_size
        )
        self.seed = seed
        self.sigma_accepatnce_threshold = sigma_accepatnce_threshold
        self.perturbations = perturbations
        self.perturbation_array = pertrubation_array
        
        self.initialize_hash_vector_dictionary()
        self.set_acceptance_threshold()

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

    def set_acceptance_threshold(self) -> None:
        if self.sigma_accepatnce_threshold < 1 or self.sigma_accepatnce_threshold > 4 or not isinstance(self.sigma_accepatnce_threshold, int):
            raise ValueError("Sigma acceptance threshold must be an integer between 1 and 4 inclusive.")
        sigma_dict = {
            1: 0.682689492137086,
            2: 0.954499736103642,
            3: 0.997300203936740,
            4: 0.999936657516333,
        }
        if 1/(1-sigma_dict[self.sigma_accepatnce_threshold]) > self.perturbations and self.sigma_accepatnce_threshold > 1:
            warnings.warn(f"Sigma acceptance threshold of {self.sigma_accepatnce_threshold} corresponds to an acceptance threshold of {sigma_dict[self.sigma_accepatnce_threshold]} which may be too high for the number of perturbations ({self.perturbations}) and could lead to false positives. Consider lowering the sigma acceptance threshold or increasing the number of perturbations to over {int(np.ceil(1/(1-sigma_dict[self.sigma_accepatnce_threshold])))}.")
            
        self.acceptance_threshold = sigma_dict[self.sigma_accepatnce_threshold]

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

    def initialize_hash_vector_dictionary(self) -> None:
        self.hash_dict = {i: {} for i in range(self.perturbations)}

    def compute_hash_vector_dictionary(self) -> dict:
        self.initialize_hash_vector_dictionary()
        for i in range(self.vector_count):
            self.add_input_vector_hashes_to_dictionary(input_vector=self._dataset_array[i], input_index=i)
        
        return self.hash_dict
    
    def add_input_vector_hashes_to_dictionary(self, input_vector: np.ndarray | None = None, input_index: int = None) -> None:
        if input_vector is None:
            input_vector = self.input_vector
        if input_index is None:
            input_index = self.vector_count

        hash_vector = self.create_hash_vector(input_vector)
        self.warn_if_vector_dict_mismatch(hash_vector)

        for hash_index, hash_value in enumerate(hash_vector):
            if hash_value not in self.hash_dict[hash_index]:
                self.hash_dict[hash_index][hash_value] = [input_index]
            else:
                self.hash_dict[hash_index][hash_value].append(input_index) # maybe replace with pre init boolean np array 

    def warn_if_vector_dict_mismatch(self, hash_vector: np.ndarray):
        if len(self.hash_dict) != len(hash_vector):
            warnings.warn("Hash vector dictionary length does not match number of perturbations; recomputing. \nAssign the perturbation array and compute the hash vector dictionary prior to duplication checks to avoid this warning.")
            self.compute_hash_vector_dictionary()

    def duplicate_check(self) -> bool:
        """duplicate check against dataset using multi-hashing approach.

        Returns:
            bool: True if a duplicate is detected, False otherwise.
        """
        hash_vector = self.create_hash_vector()
        self.warn_if_vector_dict_mismatch(hash_vector)

        clash_vector = np.zeros(self.perturbations, dtype=bool)
        for hash_index, hash_value in enumerate(hash_vector):
            if hash_value in self.hash_dict[hash_index]:
                clash_vector[hash_index] = True
        duplicate_vote_array = np.sum(clash_vector) / self.perturbations

        duplicate_structure = duplicate_vote_array >= self.acceptance_threshold

        return bool(duplicate_structure)
    
    def get_uniqueness_score(self) -> float:
        hash_vector = self.create_hash_vector()
        self.warn_if_vector_dict_mismatch(hash_vector)
        clash_vector = np.zeros(self.perturbations, dtype=bool)
        for hash_index, hash_value in enumerate(hash_vector):
            if hash_value in self.hash_dict[hash_index]:
                clash_vector[hash_index] = True
        uniqueness_score = 1 - np.sum(clash_vector) / self.perturbations

        return uniqueness_score


    def get_dataset_unique_structures(self) -> int:
        """Counts the number of unique structures in the dataset by checking for hash value clashes across the dataset for each perturbation.

        Returns:
            int: The number of unique structures in the dataset.
        """
        self.compute_hash_vector_dictionary()
        u_list_of_lists = []
        for i in range(self.perturbations):
            u_list = [self.hash_dict[i][hash_value][0] for hash_value in self.hash_dict[i]]
            u_list_of_lists.append(u_list)

        # count the number of times each index appears across all perturbations
        index_counts = {}
        for i in range(self.vector_count):
            for u_list in u_list_of_lists:
                if i in u_list:
                    if i not in index_counts:
                        index_counts[i] = 1
                    else:
                        index_counts[i] += 1
        
        # get indexes that appear in at least acceptance_threshold * perturbations perturbations
        index_filter = [index for index, count in index_counts.items() if count / self.perturbations >= self.acceptance_threshold]

        # convert to boolean array for indexing
        self.unique_vector_indices = np.zeros(self.vector_count, dtype=bool)
        self.unique_vector_indices[index_filter] = True
    
        return np.sum(self.unique_vector_indices)
    
    def pre_dda_processing(self, *args, **kwargs) -> None:
        self.set_perturbation_array()
        self.compute_hash_vector_dictionary()
        

    def add_input_vector_to_dda(self) -> None:
        """Add the input vector to the dataset array and update the hash vector array accordingly."""
        self._dataset_array[self.vector_count] = self.input_vector
        self.add_input_vector_hashes_to_dictionary(self.input_vector)
        self.vector_count += 1