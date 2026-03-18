# tests/conftest.py
import pytest
import numpy as np
from deduplicate.core.duplicate_detection_algorithm import DuplicateDetectionAlgorithm

# --- Dummy plugins for testing only ---

class DummyDDA(DuplicateDetectionAlgorithm):
    def duplicate_check(self) -> bool:
        return True
    
    def get_dataset_unique_structures(self) -> int:
        return 1

# --- Fixtures ---

@pytest.fixture
def dummy_dda(): return DummyDDA(tolerance=0.1, input_vector=np.array([1.0, 2.0]), dataset_array=np.array([[1.0, 2.0]]))


