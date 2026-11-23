"""Hashes added to a MultiHashing table since its last merge / index save.

They must answer "seen before?" for the vectors added after them, and are merged
into the sorted table (or written as an index segment) in one go later.  A
Python set costs ~60-100 bytes per hash, which limits how many vectors can be
added between saves; :class:`PendingHashes` keeps them in sorted uint64 arrays
(8 bytes per hash) instead:

* hashes added in bulk (``add_sorted``) are kept as sorted arrays, merged
  geometrically (an array absorbs the one before it while that one is at most
  ``MERGE_RATIO`` times its size), so there are O(log n) of them;
* hashes added one at a time (``add``) go into a small set first, which is
  turned into an array once it holds ``SET_SPILL`` hashes.

No hash is ever added twice (callers only add unseen hashes), so the arrays
never share a hash.
"""

import numpy as np

SET_SPILL = 4096
MERGE_RATIO = 2


def _merge_disjoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sorted union of two sorted arrays with no hash in common."""
    if len(a) < len(b):
        a, b = b, a
    return np.insert(a, np.searchsorted(a, b), b)


class PendingHashes:
    """Distinct uint64 hashes in a small set plus a few sorted arrays."""

    def __init__(self) -> None:
        self._set: set[int] = set()
        self._arrays: list[np.ndarray] = []
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    @property
    def nbytes(self) -> int:
        """Approximate memory held (arrays, plus ~100 bytes per set entry)."""
        return sum(a.nbytes for a in self._arrays) + 100 * len(self._set)

    def add(self, value: int) -> None:
        """Add one hash not already pending."""
        self._set.add(int(value))
        self._size += 1
        if len(self._set) >= SET_SPILL:
            self._push(
                np.sort(np.fromiter(self._set, dtype=np.uint64, count=len(self._set)))
            )
            self._set.clear()

    def add_sorted(self, values: np.ndarray) -> None:
        """Add sorted distinct hashes, none of them already pending."""
        if len(values):
            self._size += len(values)
            self._push(np.array(values, dtype=np.uint64))

    def _push(self, array: np.ndarray) -> None:
        while self._arrays and len(self._arrays[-1]) <= MERGE_RATIO * len(array):
            array = _merge_disjoint(self._arrays.pop(), array)
        self._arrays.append(array)

    def contains(self, value) -> bool:
        if int(value) in self._set:
            return True
        value = np.uint64(value)
        for array in self._arrays:
            position = np.searchsorted(array, value)
            if position < len(array) and array[position] == value:
                return True
        return False

    def contains_many(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.uint64)
        seen = np.zeros(values.shape, dtype=bool)
        for array in self._arrays:
            positions = np.minimum(np.searchsorted(array, values), len(array) - 1)
            seen |= array[positions] == values
        if self._set:
            seen |= np.isin(
                values, np.fromiter(self._set, dtype=np.uint64, count=len(self._set))
            )
        return seen

    def sorted(self) -> np.ndarray:
        """All pending hashes as one sorted array."""
        merged = np.sort(np.fromiter(self._set, dtype=np.uint64, count=len(self._set)))
        for array in self._arrays:
            merged = _merge_disjoint(array, merged)
        return merged

    def clear(self) -> None:
        self._set.clear()
        self._arrays = []
        self._size = 0
