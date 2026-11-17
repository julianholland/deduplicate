"""Memory budgets, chunk sizing, thread counts and progress reporting."""

import logging
import os
import re
import warnings
from collections.abc import Iterable, Iterator
from typing import TypeVar

import numba

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Used when the available memory cannot be read from the OS.
FALLBACK_MEMORY_LIMIT = 4 * 1024**3
# Fraction of the available memory "auto" claims.
AUTO_MEMORY_FRACTION = 0.5
# Shares of the memory limit given to the vectors being read and to the hashes
# being built. The rest is headroom for np.unique's sort and Python overhead.
CHUNK_FRACTION = 0.25
HASH_FRACTION = 0.5

_UNITS = {
    "": 1,
    "B": 1,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "TIB": 1024**4,
}


def _meminfo_available(path: str = "/proc/meminfo") -> int | None:
    """``MemAvailable`` from ``/proc/meminfo`` in bytes (Linux), else ``None``."""
    try:
        with open(path) as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def available_memory() -> int | None:
    """Bytes of physical memory currently available, or ``None`` if unknown.

    On Linux this is ``MemAvailable``, which counts reclaimable page cache;
    elsewhere it falls back to the free physical pages.
    """
    available = _meminfo_available()
    if available is not None:
        return available
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None


def parse_memory_limit(memory_limit: int | float | str | None) -> int:
    """Convert a memory limit to bytes.

    Accepts a number of bytes, a string with a unit (``"512MB"``, ``"8GB"``,
    ``"8GiB"``; decimal and binary units), or ``"auto"``/``None`` for half of
    the currently available memory.
    """
    if memory_limit is None or (
        isinstance(memory_limit, str) and memory_limit.strip().lower() == "auto"
    ):
        available = available_memory()
        if available is None:
            logger.info(
                "Could not read the available memory; using a memory limit of %d bytes.",
                FALLBACK_MEMORY_LIMIT,
            )
            return FALLBACK_MEMORY_LIMIT
        return max(1, int(available * AUTO_MEMORY_FRACTION))

    if isinstance(memory_limit, bool):
        raise TypeError("memory_limit must be a number of bytes, a string or 'auto'.")
    if isinstance(memory_limit, (int, float)):
        limit = int(memory_limit)
    elif isinstance(memory_limit, str):
        match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)\s*", memory_limit)
        if match is None or match.group(2).upper() not in _UNITS:
            raise ValueError(
                f"Cannot parse memory_limit {memory_limit!r}; use e.g. '512MB', "
                "'8GB', '8GiB', a number of bytes, or 'auto'."
            )
        limit = int(float(match.group(1)) * _UNITS[match.group(2).upper()])
    else:
        raise TypeError("memory_limit must be a number of bytes, a string or 'auto'.")

    if limit <= 0:
        raise ValueError("memory_limit must be positive.")
    return limit


def chunk_rows(dim: int, memory_limit: int) -> int:
    """Rows per chunk read from a store, counting each value as a float64."""
    return max(1, int(memory_limit * CHUNK_FRACTION) // (8 * max(1, dim)))


def perturbation_batch(n_vectors: int, perturbations: int, memory_limit: int) -> int:
    """Perturbations whose hashes (``uint64`` per vector) are built in one pass over the data."""
    per_perturbation = 8 * max(1, n_vectors)
    batch = int(memory_limit * HASH_FRACTION) // per_perturbation
    return max(1, min(perturbations, batch))


def set_threads(n_threads: int | None) -> None:
    """Set the number of threads numba's parallel kernels use (``None`` keeps the current setting)."""
    if n_threads is None:
        return
    if not isinstance(n_threads, int) or isinstance(n_threads, bool) or n_threads < 1:
        raise ValueError("n_threads must be a positive integer or None.")
    numba.set_num_threads(min(n_threads, numba.config.NUMBA_NUM_THREADS))


def progress_bar(
    iterable: Iterable[T], enabled: bool, total: int | None = None, desc: str = ""
) -> Iterator[T]:
    """Wrap ``iterable`` in a tqdm progress bar if enabled and tqdm is installed."""
    if not enabled:
        return iter(iterable)
    try:
        from tqdm.auto import tqdm
    except ImportError:
        warnings.warn(
            "progress=True needs tqdm; install it with `pip install deduplicate_lib[progress]`.",
            stacklevel=3,
        )
        return iter(iterable)
    return iter(tqdm(iterable, total=total, desc=desc))
