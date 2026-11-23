import mmap

import numpy as np
import pytest

from deduplicate_lib.core.sorted_segment import (
    FENCE_STRIDE,
    ITEM_BYTES,
    PAGE_BYTES,
    SortedSegment,
    block_starts,
)

SIZES = [
    0,
    1,
    2,
    FENCE_STRIDE - 1,
    FENCE_STRIDE,
    FENCE_STRIDE + 1,
    5 * FENCE_STRIDE + 17,
]


def sorted_table(n, seed=0):
    rng = np.random.default_rng(seed)
    values = np.unique(rng.integers(10, 2**63, size=2 * n + 10, dtype=np.uint64))
    return np.sort(rng.choice(values, size=n, replace=False)) if n else values[:0]


def queries_for(table, seed=1):
    rng = np.random.default_rng(seed)
    misses = rng.integers(0, 2**64 - 1, size=200, dtype=np.uint64)
    edges = np.array([0, 1, 2**64 - 1], dtype=np.uint64)
    return np.concatenate([table, misses, edges, table[:: max(1, len(table) // 7)] + 1])


def expected(table, queries):
    if not len(table):
        return np.zeros(queries.shape, dtype=bool)
    positions = np.minimum(np.searchsorted(table, queries), len(table) - 1)
    return table[positions] == queries


@pytest.mark.parametrize("n", SIZES)
def test_contains_matches_searchsorted_in_memory(n):
    table = sorted_table(n)
    segment = SortedSegment(table)
    queries = queries_for(table)
    assert np.array_equal(segment.contains_many(queries), expected(table, queries))
    for q in queries[:20]:
        assert segment.contains(q) == bool(expected(table, np.array([q]))[0])
    segment.prefetch(queries)  # no-op in memory


@pytest.mark.parametrize("n", SIZES)
def test_contains_matches_searchsorted_memory_mapped(tmp_path, n):
    table = sorted_table(n, seed=n)
    np.save(tmp_path / "t.npy", table)
    mapped = np.load(tmp_path / "t.npy", mmap_mode="r")
    segment = SortedSegment(mapped)
    queries = queries_for(table)
    assert np.array_equal(segment.contains_many(queries), expected(table, queries))
    segment.prefetch(queries)
    segment.prefetch(queries[0])


@pytest.mark.parametrize("offset", [0, 128, 4096, 4000])
@pytest.mark.parametrize("n", SIZES)
def test_blocks_are_page_aligned(n, offset):
    starts = block_starts(n, offset)
    assert len(starts) == (0 if n == 0 else len(starts))
    if n:
        assert starts[0] == 0 and np.all(np.diff(starts) > 0) and starts[-1] < n
        for start in starts[1:]:
            assert (offset + start * ITEM_BYTES) % PAGE_BYTES == 0
        assert np.all(np.diff(starts) <= FENCE_STRIDE)


def test_fence_for_matches_fence_of_the_mapped_file(tmp_path):
    table = sorted_table(3 * FENCE_STRIDE + 5)
    np.save(tmp_path / "t.npy", table)
    mapped = np.load(tmp_path / "t.npy", mmap_mode="r")
    fence = SortedSegment.fence_for(table, mapped.offset)
    assert np.array_equal(fence, SortedSegment(mapped).fence)
    # a stale fence (wrong length) is recomputed
    assert np.array_equal(SortedSegment(mapped, fence[:1]).fence, fence)


def test_fence_is_small():
    table = sorted_table(100 * FENCE_STRIDE)
    assert (
        SortedSegment(table).fence.nbytes <= table.nbytes // FENCE_STRIDE + ITEM_BYTES
    )


def test_segment_is_array_like():
    table = sorted_table(10)
    segment = SortedSegment(table)
    assert len(segment) == 10 and segment.shape == (10,) and segment.nbytes == 80
    assert np.array_equal(np.asarray(segment), table)
    assert np.array_equal(np.sort(segment), table)


def test_page_size_is_known():
    assert PAGE_BYTES == mmap.PAGESIZE and FENCE_STRIDE * ITEM_BYTES == PAGE_BYTES
