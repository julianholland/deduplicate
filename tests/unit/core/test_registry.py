# tests/test_registry.py
import pytest
from deduplicate_lib.core.plugin_registry import register_plugin, create_plugin, get_plugin_class
from deduplicate_lib.core.duplicate_detection_algorithm import DuplicateDetectionAlgorithm

def test_register_and_retrieve():
    @register_plugin("duplicate_detection_algorithm", "test_dummy")
    class _Dummy(DuplicateDetectionAlgorithm):
        def duplicate_check(self): return True
        def get_dataset_unique_structures(self): return 1

    cls = get_plugin_class("duplicate_detection_algorithm", "test_dummy")
    assert cls is _Dummy

def test_create_plugin():
    kwargs_dict = {
        "tolerance": 0.1,
        "input_vector": [1.0, 2.0],
        "dataset_array": [[1.0, 2.0]],
    }
    instance = create_plugin("duplicate_detection_algorithm", "test_dummy", **kwargs_dict)
    assert isinstance(instance, DuplicateDetectionAlgorithm)

def test_missing_plugin_raises():
    with pytest.raises(KeyError, match="nonexistent"):
        get_plugin_class("duplicate_detection_algorithm", "nonexistent")

def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        @register_plugin("invalid_kind", "x")
        class _Bad:
            pass
