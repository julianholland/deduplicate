# tests/conftest.py
import pytest
import numpy as np
from deduplicate_lib.core.duplicate_detection_algorithm import DuplicateDetectionAlgorithm
from deduplicate_lib.core.tolerance_calculator import ToleranceCalculator
from deduplicate_lib.plugins.duplicate_detection_algorithms.distance_matrix import DistanceMatrix
from deduplicate_lib.plugins.duplicate_detection_algorithms.multi_hashing import MultiHashing

# --- Dummy plugins for testing only ---


class DummyDDA(DuplicateDetectionAlgorithm):
    def duplicate_check(self) -> bool:
        return True

    def get_dataset_unique_structures(self) -> int:
        return 1


class DummyToleranceCalculator(ToleranceCalculator):
    def calculate_tolerance(self) -> float:
        return 0.1


# --- Fixtures ---


dummy_dda_object = DummyDDA(
    tolerance=0.1,
    input_vector=np.array([1.0, 2.0]),
    dataset_array=np.array([[1.0, 2.0]]),
)


@pytest.fixture
def dummy_dda():
    return dummy_dda_object


@pytest.fixture
def dummy_tolerance_calculator():
    return DummyToleranceCalculator(
        duplicate_detection_algorithm_object=dummy_dda_object,
        tolerance_dataset_array=np.array([]),
        perturbations_per_vector=5,
        perturbation_scale=0.1,
        binary_search_steps=20,
    )

@pytest.fixture
def distance_matrix_dda():
    return DistanceMatrix(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[0.1, 0.2], [1.01, 2.01], [0.0, 20.0]]),
    )


@pytest.fixture
def multi_hashing_dda():
    return MultiHashing(
        tolerance=0.01,
        perturbations=5,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[0.1, 0.2], [0.5, 0.6], [1.8, 0.9]]),
    )
