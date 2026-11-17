import pytest

from deduplicate_lib.core import hash_index


@pytest.fixture(autouse=True)
def no_fsync(monkeypatch):
    """fsync makes no difference to what the tests check, and is slow on some filesystems."""
    monkeypatch.setattr(hash_index, "FSYNC", False)
