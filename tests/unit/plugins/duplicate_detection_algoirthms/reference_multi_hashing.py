"""The dict-based MultiHashing algorithm of deduplicate_lib <= 0.1.3, kept as a
test-only reference that the flat-array implementation must agree with."""

import numpy as np


def _reference_hash_vector(vector, perturbation_array, tolerance):
    rounded = np.array(
        [np.round(p * vector / tolerance) * tolerance for p in perturbation_array]
    )
    rounded = np.where(rounded == 0, 0.0, rounded)
    return [hash(row.tobytes()) for row in rounded]


def reference_hash_dict(dataset, perturbation_array, tolerance):
    """``{perturbation: {hash: {index: is_first_in_bucket}}}``, built in dataset order."""
    hash_dict = {i: {} for i in range(len(perturbation_array))}
    for index, vector in enumerate(dataset):
        for hash_index, hash_value in enumerate(
            _reference_hash_vector(vector, perturbation_array, tolerance)
        ):
            bucket = hash_dict[hash_index].setdefault(hash_value, {})
            bucket[index] = len(bucket) == 0
    return hash_dict


def reference_unique_mask(dataset, perturbation_array, tolerance, acceptance_threshold):
    hash_dict = reference_hash_dict(dataset, perturbation_array, tolerance)
    counts = np.zeros(len(dataset), dtype=int)
    for buckets in hash_dict.values():
        for bucket in buckets.values():
            for index, is_first in bucket.items():
                counts[index] += is_first
    return counts / len(perturbation_array) >= acceptance_threshold


def reference_clash_vector(dataset, vector, perturbation_array, tolerance):
    hash_dict = reference_hash_dict(dataset, perturbation_array, tolerance)
    return np.array(
        [
            hash_value in hash_dict[hash_index]
            for hash_index, hash_value in enumerate(
                _reference_hash_vector(vector, perturbation_array, tolerance)
            )
        ]
    )
