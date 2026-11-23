import numpy as np

from deduplicate_lib.core import pending_hashes
from deduplicate_lib.core.pending_hashes import PendingHashes


def test_matches_a_set(monkeypatch):
    monkeypatch.setattr(pending_hashes, "SET_SPILL", 16)
    rng = np.random.default_rng(0)
    values = rng.permutation(rng.integers(1, 2**63, size=5000, dtype=np.uint64))
    values = np.unique(values)
    rng.shuffle(values)
    pending, reference = PendingHashes(), set()
    position = 0
    while position < len(values):
        if rng.random() < 0.5:  # a bulk batch
            batch = np.sort(values[position : position + rng.integers(1, 300)])
            pending.add_sorted(batch)
            reference.update(batch.tolist())
            position += len(batch)
        else:  # single additions
            for value in values[position : position + 5]:
                pending.add(int(value))
                reference.add(int(value))
            position += 5
        assert len(pending) == len(reference)
    assert len(pending._arrays) <= 2 * np.log2(len(values))  # merged geometrically
    assert np.array_equal(
        pending.sorted(), np.array(sorted(reference), dtype=np.uint64)
    )
    queries = np.concatenate(
        [values[::7], rng.integers(1, 2**63, 500, dtype=np.uint64)]
    )
    expected = np.array([int(q) in reference for q in queries])
    assert np.array_equal(pending.contains_many(queries), expected)
    assert [pending.contains(q) for q in queries[:50]] == expected[:50].tolist()


def test_set_spills_into_an_array(monkeypatch):
    monkeypatch.setattr(pending_hashes, "SET_SPILL", 4)
    pending = PendingHashes()
    for value in (5, 3, 9, 1):
        pending.add(value)
    assert not pending._set and len(pending._arrays) == 1
    assert pending.sorted().tolist() == [1, 3, 5, 9]
    assert pending.nbytes == 32


def test_empty_and_clear():
    pending = PendingHashes()
    assert not pending and len(pending) == 0
    assert pending.sorted().dtype == np.uint64 and len(pending.sorted()) == 0
    assert not pending.contains_many(np.array([1, 2], dtype=np.uint64)).any()
    pending.add_sorted(np.array([1, 2], dtype=np.uint64))
    pending.add_sorted(np.array([], dtype=np.uint64))
    assert pending and pending.contains(2)
    pending.clear()
    assert not pending and not pending.contains(2)
