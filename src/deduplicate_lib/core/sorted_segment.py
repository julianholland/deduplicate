"""Sorted hash segments with an in-memory fence, for cheap lookups in memory-mapped tables.

A plain binary search (``np.searchsorted``) over a memory-mapped table of n
hashes touches about log2(n) pages, and once the table no longer fits in the
page cache most of them are cold: a lookup then costs a dozen or more disk
reads, each pulling in the kernel's mmap read-around.

A :class:`SortedSegment` keeps a **fence** in memory: the first hash of every
file page of the table (``FENCE_STRIDE`` hashes, ~1/512 of the table).  A
lookup searches the fence in memory, then only the one page that can hold the
hash, so a cold lookup costs exactly one page read.  :meth:`SortedSegment.prefetch`
asks the kernel for those pages up front (``MADV_WILLNEED``), so the lookups of
many hashes (e.g. one per perturbation of a vector) read their pages
concurrently instead of one fault at a time.
"""

import contextlib
import mmap

import numpy as np
from numba import njit

PAGE_BYTES = mmap.PAGESIZE
ITEM_BYTES = np.dtype(np.uint64).itemsize
# hashes per fence entry: one file page
FENCE_STRIDE = max(1, PAGE_BYTES // ITEM_BYTES)


def block_starts(n: int, data_offset: int = 0) -> np.ndarray:
    """First row of each block of a table of ``n`` hashes whose data starts
    ``data_offset`` bytes into its file: blocks are aligned to file pages, so
    each one (but the first, if the data is unaligned) is exactly one page."""
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    first = (
        (-data_offset) % PAGE_BYTES
    ) // ITEM_BYTES  # rows before the next page boundary
    starts = np.arange(first, n, FENCE_STRIDE, dtype=np.int64)
    if first:
        starts = np.concatenate(([0], starts))
    return starts


@njit(cache=True)
def _block_of(fence, value):  # pragma: no cover
    """Index of the last fence entry <= value, or -1."""
    lo, hi = 0, fence.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if fence[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


@njit(cache=True)
def fenced_contains(table, fence, starts, queries):  # pragma: no cover
    """Whether each of ``queries`` is in the sorted ``table``, searching only
    the block the fence points at."""
    n = table.shape[0]
    found = np.zeros(queries.shape[0], dtype=np.bool_)
    for i in range(queries.shape[0]):
        value = queries[i]
        block = _block_of(fence, value)
        if block < 0:
            continue
        lo = starts[block]
        hi = starts[block + 1] if block + 1 < starts.shape[0] else n
        while lo < hi:
            mid = (lo + hi) // 2
            if table[mid] < value:
                lo = mid + 1
            else:
                hi = mid
        if lo < n and table[lo] == value:
            found[i] = True
    return found


class SortedSegment:
    """A sorted array of distinct uint64 hashes (in memory or memory-mapped)
    with an in-memory fence: the first hash of each file page."""

    def __init__(self, table: np.ndarray, fence: np.ndarray | None = None) -> None:
        self.table = table
        self._data_offset = int(getattr(table, "offset", 0) or 0)
        self.starts = block_starts(table.shape[0], self._data_offset)
        if fence is None or fence.shape[0] != self.starts.shape[0]:
            fence = np.ascontiguousarray(
                np.asarray(table)[self.starts], dtype=np.uint64
            )
        self.fence = fence
        self._mmap = getattr(table, "_mmap", None)
        self._map_start = (
            self._data_offset - self._data_offset % mmap.ALLOCATIONGRANULARITY
        )

    @staticmethod
    def fence_for(table: np.ndarray, data_offset: int) -> np.ndarray:
        """The fence of ``table`` once it is stored ``data_offset`` bytes into a file."""
        return np.ascontiguousarray(
            np.asarray(table)[block_starts(table.shape[0], data_offset)],
            dtype=np.uint64,
        )

    # array-like, so existing code can treat a segment as its table
    def __len__(self) -> int:
        return self.table.shape[0]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.table.shape

    @property
    def nbytes(self) -> int:
        """Bytes of the table (the fence is ~1/FENCE_STRIDE of it on top)."""
        return self.table.nbytes

    def __array__(self, dtype=None, copy=None):
        array = np.asarray(self.table)
        return array if dtype is None else array.astype(dtype, copy=False)

    def contains_many(self, values: np.ndarray) -> np.ndarray:
        values = np.ascontiguousarray(values, dtype=np.uint64)
        if not self.table.shape[0]:
            return np.zeros(values.shape, dtype=bool)
        return fenced_contains(np.asarray(self.table), self.fence, self.starts, values)

    def contains(self, value) -> bool:
        return bool(self.contains_many(np.array([value], dtype=np.uint64))[0])

    def prefetch(self, values: np.ndarray) -> None:
        """Ask the kernel to read the pages that may hold ``values`` (memory-mapped
        tables only), without waiting for them."""
        if (
            self._mmap is None
            or not self.table.shape[0]
            or not hasattr(self._mmap, "madvise")
        ):
            return
        blocks = np.searchsorted(
            self.fence, np.atleast_1d(np.asarray(values, dtype=np.uint64)), side="right"
        )
        for block in {int(b) - 1 for b in blocks.tolist() if b > 0}:
            start = int(self.starts[block])
            stop = (
                int(self.starts[block + 1])
                if block + 1 < self.starts.shape[0]
                else self.table.shape[0]
            )
            first = self._data_offset + start * ITEM_BYTES - self._map_start
            last = self._data_offset + stop * ITEM_BYTES - self._map_start
            page = first - first % PAGE_BYTES
            with contextlib.suppress(OSError, ValueError):
                self._mmap.madvise(mmap.MADV_WILLNEED, page, last - page)
