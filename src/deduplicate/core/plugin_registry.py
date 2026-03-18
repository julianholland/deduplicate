# core/plugin_registry.py
from typing import Dict, Type

_REGISTRY: Dict[str, Dict[str, Type]] = {
    "duplicate_detection_algorithm": {},
}


def register_plugin(kind: str, name: str):
    def decorator(cls):
        if kind not in _REGISTRY:
            raise ValueError(f"Unknown plugin kind: {kind}")
        _REGISTRY[kind][name] = cls
        return cls

    return decorator


def get_plugin_class(kind: str, name: str):
    try:
        return _REGISTRY[kind][name]
    except KeyError as e:
        raise KeyError(f"No plugin '{name}' registered for kind '{kind}'") from e


def create_plugin(kind: str, name: str, **kwargs):
    cls = get_plugin_class(kind, name)
    return cls(**kwargs)
