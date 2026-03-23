# tests/conftest.py
import pytest
import numpy as np
from deduplicate.core.duplicate_detection_algorithm import DuplicateDetectionAlgorithm
from deduplicate.core.tolerance_calculator import ToleranceCalculator

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
