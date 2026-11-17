import logging
import warnings
from pathlib import Path
from typing import ClassVar

import numpy as np
from numba import njit, prange

from deduplicate_lib.core import resources
from deduplicate_lib.core.duplicate_detection_algorithm import (
    DuplicateDetectionAlgorithm,
    hamming_distance,
)
from deduplicate_lib.core.hash_index import HashIndex, perturbation_digest
from deduplicate_lib.core.plugin_registry import register_plugin
from deduplicate_lib.core.vector_store import InMemoryStore, VectorStore

logger = logging.getLogger(__name__)

# splitmix64 constants. Shift amounts are uint64 too: numba promotes mixed
# uint64/int64 arithmetic to float64.
_HASH_SEED = np.uint64(0x9E3779B97F4A7C15)
_MIX_C1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_C2 = np.uint64(0x94D049BB133111EB)
_SHIFT_30 = np.uint64(30)
_SHIFT_27 = np.uint64(27)
_SHIFT_31 = np.uint64(31)

# Hashes persisted from one version are only comparable with hashes from the
# same version; bump when fast_quantise_and_hash changes.
HASH_VERSION = 1


@njit
def _mix64(z):
    z = (z ^ (z >> _SHIFT_30)) * _MIX_C1  # pragma: no cover
    z = (z ^ (z >> _SHIFT_27)) * _MIX_C2  # pragma: no cover
    return z ^ (z >> _SHIFT_31)  # pragma: no cover


@njit
def _hash_vector(vector, perturbation, tolerance, grid, grid_bits):
    for j in range(vector.shape[0]):  # pragma: no cover
        v = perturbation * vector[j] / tolerance  # pragma: no cover
        # + 0.0 turns -0.0 into 0.0, which would otherwise hash differently
        grid[j] = np.rint(v) + 0.0  # pragma: no cover
    h = _HASH_SEED  # pragma: no cover
    for j in range(vector.shape[0]):  # pragma: no cover
        h = _mix64(h ^ grid_bits[j])  # pragma: no cover
    return h  # pragma: no cover


@njit(parallel=True)
def fast_quantise_and_hash(vector_array, perturbation_array, tolerance):
    """Hash every vector under every perturbation.

    Each vector is scaled by the perturbation, rounded to the ``tolerance``
    grid and hashed, so two vectors share a hash exactly when they round to the
    same grid point. Hashes are deterministic across processes (unlike
    Python's ``hash``), so they can be persisted and compared later.

    Returns:
        np.ndarray: ``uint64`` array of shape ``(len(perturbation_array), len(vector_array))``.
    """
    n_pert = perturbation_array.shape[0]  # pragma: no cover
    n_vec = vector_array.shape[0]  # pragma: no cover
    n_feat = vector_array.shape[1]  # pragma: no cover
    out = np.empty((n_pert, n_vec), dtype=np.uint64)  # pragma: no cover

    for k in prange(n_vec):  # pragma: no cover
        grid = np.empty(n_feat, dtype=np.float64)  # pragma: no cover
        grid_bits = grid.view(np.uint64)  # pragma: no cover
        for i in range(n_pert):  # pragma: no cover
            out[i, k] = _hash_vector(
                vector_array[k], perturbation_array[i], tolerance, grid, grid_bits
            )  # pragma: no cover

    return out  # pragma: no cover


@njit
def fast_hash_single_vector(vector, perturbation_array, tolerance):
    """Hashes of one vector under every perturbation; the serial counterpart of
    :func:`fast_quantise_and_hash` for single vectors, where threads cost more
    than they save."""
    out = np.empty(perturbation_array.shape[0], dtype=np.uint64)  # pragma: no cover
    grid = np.empty(vector.shape[0], dtype=np.float64)  # pragma: no cover
    grid_bits = grid.view(np.uint64)  # pragma: no cover
    for i in range(perturbation_array.shape[0]):  # pragma: no cover
        out[i] = _hash_vector(
            vector, perturbation_array[i], tolerance, grid, grid_bits
        )  # pragma: no cover
    return out  # pragma: no cover


@register_plugin(kind="duplicate_detection_algorithm", name="multi_hashing")
class MultiHashing(DuplicateDetectionAlgorithm):
    """Approximate deduplication via random perturbation and hashing.

    Each vector is smeared with ``perturbations`` random scale factors, rounded
    to the nearest ``tolerance`` grid point, and hashed.  For every
    perturbation, the earliest vector in each hash bucket counts as unique for
    that perturbation; a vector is unique overall if it is unique in at least
    ``acceptance_threshold`` (derived from ``sigma_accepatnce_threshold``) of
    the perturbations.  ``duplicate_check()`` declares the input vector a
    duplicate if at least ``acceptance_threshold`` of its hashes collide with
    the dataset's.  O(N · perturbations) time; significantly faster than
    :class:`DistanceMatrix` for large datasets.

    The hash tables are flat ``uint64`` arrays, one sorted array of the
    dataset's distinct hashes per perturbation, plus a per-vector count of the
    perturbations in which it is unique.  They are built from the dataset in
    chunks and perturbation batches sized by ``memory_limit``, so an
    out-of-core ``dataset`` (see :class:`DuplicateDetectionAlgorithm`) is
    never loaded whole.  If the per-perturbation hash arrays themselves would
    exceed a quarter of ``memory_limit``, only the unique counts are kept:
    ``get_dataset_unique_structures()`` and ``deduplicate()`` still work, but
    ``duplicate_check()``, ``get_uniqueness_score()`` and appending vectors
    need the tables: pass ``index_dir`` to keep them on disk instead.

    With ``index_dir``, the tables are saved to that directory (see
    :mod:`deduplicate_lib.core.hash_index`) and memory-mapped, so they never
    have to fit in memory.  Reopening the same dataset with the same
    ``tolerance`` and perturbations reuses them instead of hashing again; an
    interrupted build resumes where it stopped; and vectors appended later
    (``add_vectors()``, ``add_input_vector_to_dda()``) are saved with
    ``flush()``.  Each tolerance needs its own ``index_dir``.

    Only ``"hamming"`` distance is supported.

    Parameters
    ----------
    tolerance : float, optional
        Rounding grid size and distance threshold.  Defaults to ``0.1``.
    input_vector : np.ndarray, optional
        Single vector to check against the dataset.
    dataset_array : np.ndarray, optional
        Initial dataset of vectors (rows).
    perturbations : int, optional
        Number of random perturbations per vector.  More perturbations increase
        accuracy at the cost of speed.  Defaults to ``200``.
    seed : int, optional
        Random seed for generating the perturbation array.  Defaults to ``803``.
    pertrubation_array : np.ndarray, optional
        Pre-computed perturbation scale factors.  Generated from ``seed`` if
        not provided or if its length does not match ``perturbations``.
    distance_metric : str, optional
        Must be ``"hamming"``.
    sigma_accepatnce_threshold : int, optional
        Controls the acceptance band (1-4):

        ======  ================
        Value   Acceptance band
        ======  ================
        1       68.3%  (default)
        2       95.4%
        3       99.7%
        4       99.99%
        ======  ================

        Higher values reduce false positives but require more perturbations.
    distance_matrix : np.ndarray, optional
        Unused by this algorithm; kept for interface compatibility.
    unique_vector_indices : np.ndarray, optional
        Boolean array marking unique vectors.
    max_vector_array_size : int, optional
        Maximum number of vectors the pre-allocated arrays can hold.
        Defaults to 10000.
    dataset, memory_limit, n_threads, progress : optional
        See :class:`DuplicateDetectionAlgorithm`.
    index_dir : str or Path, optional
        Directory for persistent hash tables.  Created if missing; reused if it
        holds an index for this dataset and these parameters; otherwise
        ``IndexMismatchError`` (a ``ValueError``) explains what differs.
    """

    ALLOWED_DISTANCES: ClassVar[dict] = {"hamming": hamming_distance}

    def __init__(
        self,
        tolerance: float = 0.1,
        input_vector: np.ndarray = np.array([]),
        dataset_array: np.ndarray = np.array([]),
        perturbations: int = 200,
        seed: int = 803,
        pertrubation_array: np.ndarray = np.array([]),
        distance_metric: str = "hamming",
        sigma_accepatnce_threshold: int = 1,
        distance_matrix: np.ndarray = np.array([]),
        unique_vector_indices: np.ndarray = np.array([]),
        max_vector_array_size: int = 10000,
        dataset: str | Path | VectorStore | np.ndarray | None = None,
        memory_limit: int | float | str | None = "auto",
        n_threads: int | None = None,
        progress: bool = False,
        index_dir: str | Path | None = None,
    ) -> None:
        self.index_dir = None if index_dir is None else Path(index_dir)
        self.index: HashIndex | None = None
        super().__init__(
            tolerance=tolerance,
            input_vector=input_vector,
            dataset_array=dataset_array,
            distance_metric=distance_metric,
            distance_matrix=distance_matrix,
            unique_vector_indices=unique_vector_indices,
            max_vector_array_size=max_vector_array_size,
            dataset=dataset,
            memory_limit=memory_limit,
            n_threads=n_threads,
            progress=progress,
        )
        self.seed = seed
        self.sigma_accepatnce_threshold = sigma_accepatnce_threshold
        self.perturbations = perturbations
        self.perturbation_array = pertrubation_array

        self.initialize_hash_tables()
        self.set_acceptance_threshold()

    def __str__(self) -> str:
        return f"MultiHashing(tolerance={self.tolerance}, perturbations={self.perturbations}, sigma_acceptance_threshold={self.sigma_accepatnce_threshold})"

    def set_perturbation_array(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.perturbation_array = np.abs(
            rng.normal(loc=0.0, scale=1.0, size=self.perturbations)
        )
        # remember the seed so a seed change regenerates it; a user-supplied
        # array (no generating seed) is kept as given
        self._perturbation_seed = self.seed

    def _ensure_perturbation_array(self):

        if self.perturbation_array.shape[0] != self.perturbations:
            warnings.warn(
                "Perturbation array shape does not match expected number of perturbations; recomputing. \nAssign the perturbation array prior duplication checks to avoid this warning.",
                stacklevel=2,
            )
            self.set_perturbation_array()

    def _init_kwargs(self) -> dict:
        kwargs = super()._init_kwargs()
        kwargs.update(
            perturbations=self.perturbations,
            seed=self.seed,
            sigma_accepatnce_threshold=self.sigma_accepatnce_threshold,
        )
        if self.perturbation_array.shape[0] == self.perturbations:
            kwargs["pertrubation_array"] = self.perturbation_array.copy()
        return kwargs

    def set_acceptance_threshold(self) -> None:
        if (
            self.sigma_accepatnce_threshold < 1
            or self.sigma_accepatnce_threshold > 4
            or not isinstance(self.sigma_accepatnce_threshold, int)
        ):
            raise ValueError(
                "Sigma acceptance threshold must be an integer between 1 and 4 inclusive."
            )
        sigma_dict = {
            1: 0.682689492137086,
            2: 0.954499736103642,
            3: 0.997300203936740,
            4: 0.999936657516333,
        }
        if (
            1 / (1 - sigma_dict[self.sigma_accepatnce_threshold]) > self.perturbations
            and self.sigma_accepatnce_threshold > 1
        ):
            warnings.warn(
                f"Sigma acceptance threshold of {self.sigma_accepatnce_threshold} corresponds to an acceptance threshold of {sigma_dict[self.sigma_accepatnce_threshold]} which may be too high for the number of perturbations ({self.perturbations}) and could lead to false positives. Consider lowering the sigma acceptance threshold or increasing the number of perturbations to over {int(np.ceil(1 / (1 - sigma_dict[self.sigma_accepatnce_threshold])))}.",
                stacklevel=2,
            )

        self.acceptance_threshold = sigma_dict[self.sigma_accepatnce_threshold]

    def round_to_tolerance(self) -> np.ndarray:
        rounded_vector = np.round(self.input_vector / self.tolerance) * self.tolerance
        return rounded_vector

    def create_hash_vector(self, input_vector: np.ndarray | None = None) -> np.ndarray:
        """Hashes of one vector, one per perturbation (``uint64``, shape ``(perturbations,)``)."""
        if input_vector is None:
            input_vector = self.input_vector
        self._ensure_perturbation_array()
        return fast_hash_single_vector(
            np.ascontiguousarray(input_vector, dtype=np.float64),
            self.perturbation_array,
            self.tolerance,
        )

    def initialize_hash_tables(self) -> None:
        """Empty hash tables: no hashes per perturbation, no unique counts."""
        self.sorted_hashes = [
            np.empty(0, dtype=np.uint64) for _ in range(self.perturbations)
        ]
        # Hashes of vectors appended since the last rebuild (or index commit),
        # merged into sorted_hashes once they grow too large to scan cheaply.
        self._pending_hashes = [set() for _ in range(self.perturbations)]
        self._pending_total = 0
        # Leading dataset rows whose hashes are in the tables (incl. pending).
        self._rows_in_tables = 0
        self.unique_counts = np.zeros(
            max(self.max_vector_array_size, self.vector_count), dtype=np.uint32
        )

    def hash_tables_fit_in_memory(self, n_vectors: int | None = None) -> bool:
        """Whether the per-perturbation hash arrays for ``n_vectors`` fit in a quarter of ``memory_limit``."""
        if n_vectors is None:
            n_vectors = self.vector_count
        return 8 * self.perturbations * n_vectors <= self.memory_limit // 4

    def _index_parameters(self) -> dict:
        return {
            "hash_version": HASH_VERSION,
            "tolerance": float(self.tolerance),
            "perturbations": int(self.perturbations),
            "perturbation_sha256": perturbation_digest(self.perturbation_array),
            "seed": int(self.seed),
            "dim": int(self.store.dim),
        }

    def _hash_perturbations(
        self, store, n_vectors, perturbation_ids, on_table, on_batch
    ):
        """Hash the first ``n_vectors`` rows of ``store`` under each of ``perturbation_ids``.

        Perturbations are processed in batches sized by ``memory_limit``; each
        batch reads the rows once, chunk by chunk.  For each perturbation,
        ``np.unique(..., return_index=True)`` gives the distinct hashes and the
        earliest vector holding each, which is the vector that counts as unique
        for that perturbation: ``on_table(perturbation, distinct, first_index)``.
        ``on_batch()`` is called after each batch.
        """
        resources.set_threads(self.n_threads)
        rows_per_chunk = self.chunk_rows()
        batch = resources.perturbation_batch(
            n_vectors, len(perturbation_ids), self.memory_limit
        )
        batches = [
            perturbation_ids[i : i + batch]
            for i in range(0, len(perturbation_ids), batch)
        ]
        n_chunks = -(-n_vectors // rows_per_chunk)
        logger.info(
            "Hashing %d vectors under %d perturbations: %d batch(es) of up to %d "
            "perturbations, %d chunk(s) of up to %d rows.",
            n_vectors,
            len(perturbation_ids),
            len(batches),
            batch,
            n_chunks,
            rows_per_chunk,
        )

        for batch_number, ids in enumerate(batches):
            hashes = np.empty((len(ids), n_vectors), dtype=np.uint64)
            chunks = resources.progress_bar(
                store.iter_chunks(rows_per_chunk, stop=n_vectors),
                self.progress,
                total=n_chunks,
                desc=f"hashing (batch {batch_number + 1}/{len(batches)})",
            )
            for chunk_start, rows in chunks:
                hashes[:, chunk_start : chunk_start + len(rows)] = (
                    fast_quantise_and_hash(
                        np.ascontiguousarray(rows, dtype=np.float64),
                        self.perturbation_array[ids],
                        self.tolerance,
                    )
                )
            for offset, perturbation in enumerate(ids):
                distinct, first_index = np.unique(hashes[offset], return_index=True)
                on_table(perturbation, distinct, first_index)
            del hashes
            on_batch()

    def compute_hash_tables(self) -> None:
        """Build the hash tables for the whole dataset.

        With ``index_dir``, the tables are kept on disk and reused: see
        :meth:`_compute_indexed_hash_tables`.
        """
        self.initialize_hash_tables()
        self._ensure_perturbation_array()
        if self.index_dir is not None:
            self._compute_indexed_hash_tables()
            return

        store = self.store
        n = len(store)
        if n == 0:
            return

        keep_tables = self.hash_tables_fit_in_memory(n)
        if not keep_tables:
            self.sorted_hashes = None
            logger.warning(
                "The hash tables for %d vectors x %d perturbations (%d bytes) do not "
                "fit in a quarter of memory_limit (%d bytes); keeping only the unique "
                "counts. get_dataset_unique_structures() and deduplicate() work, but "
                "duplicate_check(), get_uniqueness_score() and appending vectors need "
                "the tables: pass index_dir=... to keep them on disk, or raise "
                "memory_limit.",
                n,
                self.perturbations,
                8 * self.perturbations * n,
                self.memory_limit,
            )

        def on_table(perturbation, distinct, first_index):
            if keep_tables:
                self.sorted_hashes[perturbation] = distinct
            self.unique_counts[first_index] += 1

        self._hash_perturbations(
            store, n, list(range(self.perturbations)), on_table, lambda: None
        )
        self._rows_in_tables = n

    def _compute_indexed_hash_tables(self) -> None:
        """Open (or create) the index in ``index_dir`` and bring it up to date.

        * Committed tables are memory-mapped, not recomputed.
        * Perturbations still missing (a new index, or an interrupted build)
          are hashed, committing after every batch so an interrupted build
          resumes where it stopped.
        * Vectors appended to the dataset since the last commit (e.g. to an
          HDF5 store, without a ``flush()``) are added and committed.
        """
        store = self.store
        index = HashIndex.open(self.index_dir, self._index_parameters(), store)
        self.index = index
        n_indexed = index.n_vectors
        self._rows_in_tables = n_indexed
        self.unique_counts[:n_indexed] = index.load_counts()
        for perturbation in index.completed:
            self.sorted_hashes[perturbation] = index.load_table(perturbation)

        todo = [p for p in range(self.perturbations) if p not in index.completed]
        if todo:
            if len(todo) < self.perturbations:
                logger.info(
                    "Resuming the hash index: %d of %d perturbations left.",
                    len(todo),
                    self.perturbations,
                )
            staged = []

            def on_table(perturbation, distinct, first_index):
                index.write_table(perturbation, distinct)
                staged.append(perturbation)
                self.unique_counts[first_index] += 1

            def on_batch():
                index.commit(self.unique_counts[:n_indexed], store)
                for perturbation in staged:
                    self.sorted_hashes[perturbation] = index.load_table(perturbation)
                staged.clear()

            self._hash_perturbations(store, n_indexed, todo, on_table, on_batch)

        if len(store) > n_indexed:
            logger.info(
                "Adding the %d vectors appended to the dataset since the hash index "
                "was last saved.",
                len(store) - n_indexed,
            )
            for start, rows in store.iter_chunks(
                self._add_chunk_rows(), start=n_indexed
            ):
                self._add_rows_to_tables(rows, start)
            self._save_index()

    def _require_hash_tables(self, what: str) -> None:
        if self.sorted_hashes is None:
            raise MemoryError(
                f"{what} needs the hash tables, which were not kept because "
                f"{self.perturbations} perturbations x {self.vector_count} vectors "
                f"({8 * self.perturbations * self.vector_count} bytes) exceed a "
                f"quarter of memory_limit ({self.memory_limit} bytes). Pass "
                "index_dir=... to keep them on disk, or raise memory_limit."
            )

    def _hash_seen(self, hash_index: int, hash_value: np.uint64) -> bool:
        table = self.sorted_hashes[hash_index]
        position = np.searchsorted(table, hash_value)
        if position < table.shape[0] and table[position] == hash_value:
            return True
        return int(hash_value) in self._pending_hashes[hash_index]

    def _hashes_seen(self, hash_index: int, hash_values: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`_hash_seen` for many hashes of one perturbation."""
        table = self.sorted_hashes[hash_index]
        positions = np.minimum(
            np.searchsorted(table, hash_values), max(table.shape[0] - 1, 0)
        )
        seen = (
            table[positions] == hash_values
            if table.shape[0]
            else np.zeros(hash_values.shape, dtype=bool)
        )
        pending = self._pending_hashes[hash_index]
        if pending:
            seen |= np.isin(
                hash_values, np.fromiter(pending, dtype=np.uint64, count=len(pending))
            )
        return seen

    def _clash_vector(self, hash_vector: np.ndarray) -> np.ndarray:
        """Whether each hash of a vector already occurs in the dataset."""
        return np.array(
            [
                self._hash_seen(hash_index, hash_value)
                for hash_index, hash_value in enumerate(hash_vector)
            ],
            dtype=bool,
        )

    def _merged_table(self, hash_index: int) -> np.ndarray:
        """The table of ``hash_index`` with its pending hashes inserted, still sorted.

        Pending hashes are never in the table (only unseen hashes are added),
        so a sorted insert suffices: O(n), unlike re-sorting with ``union1d``.
        """
        pending = self._pending_hashes[hash_index]
        pending = np.sort(np.fromiter(pending, dtype=np.uint64, count=len(pending)))
        table = self.sorted_hashes[hash_index]
        return np.insert(np.asarray(table), np.searchsorted(table, pending), pending)

    def _merge_pending_hashes(self, hash_index: int) -> None:
        self.sorted_hashes[hash_index] = self._merged_table(hash_index)
        self._pending_total -= len(self._pending_hashes[hash_index])
        self._pending_hashes[hash_index].clear()

    def _ensure_count_capacity(self, n_vectors: int) -> None:
        if n_vectors > self.unique_counts.shape[0]:
            grown = np.zeros(
                max(2 * self.unique_counts.shape[0], n_vectors), dtype=np.uint32
            )
            grown[: self.unique_counts.shape[0]] = self.unique_counts
            self.unique_counts = grown

    def _add_chunk_rows(self) -> int:
        """Rows hashed at once when adding vectors: the rows, and their hashes
        under every perturbation, each within a quarter of ``memory_limit``."""
        return max(
            1,
            min(
                self.chunk_rows(),
                self.memory_limit // 4 // (8 * max(1, self.perturbations)),
            ),
        )

    def _pending_limit(self) -> int:
        """Pending hashes (Python ints in sets, ~100 bytes each) kept before an index flush."""
        return max(10_000, self.memory_limit // 4 // 100)

    def _after_adding_hashes(self, touched) -> None:
        """Merge pending hashes that have grown large: into the in-memory tables,
        or, with an index, into new table files via :meth:`flush`."""
        if self.index_dir is not None:
            if self._pending_total > self._pending_limit():
                self._save_index()
            return
        for hash_index in touched:
            pending = self._pending_hashes[hash_index]
            if len(pending) > max(1024, self.sorted_hashes[hash_index].shape[0] // 8):
                self._merge_pending_hashes(hash_index)

    def add_input_vector_hashes(
        self, input_vector: np.ndarray | None = None, input_index: int | None = None
    ) -> None:
        """Add one vector's hashes to the tables, counting the perturbations in which it is unique."""
        if input_vector is None:
            input_vector = self.input_vector
        if input_index is None:
            input_index = self.vector_count

        self._require_hash_tables("Adding vectors")
        hash_vector = self.create_hash_vector(input_vector)
        clash_vector = self._clash_vector(hash_vector)
        self._ensure_count_capacity(input_index + 1)
        self.unique_counts[input_index] = np.sum(~clash_vector)
        new = np.where(~clash_vector)[0]
        for hash_index in new:
            self._pending_hashes[hash_index].add(int(hash_vector[hash_index]))
        self._pending_total += len(new)
        self._rows_in_tables = max(self._rows_in_tables, input_index + 1)
        self._after_adding_hashes(new)

    def _add_rows_to_tables(self, rows: np.ndarray, start: int) -> None:
        """Add the hashes of dataset rows ``start, start + 1, ...`` to the tables.

        Equivalent to adding them one at a time: a row is unique for a
        perturbation if its hash is in neither the tables nor an earlier row.
        """
        self._require_hash_tables("Adding vectors")
        resources.set_threads(self.n_threads)
        self._ensure_count_capacity(start + len(rows))
        hashes = fast_quantise_and_hash(
            np.ascontiguousarray(rows, dtype=np.float64),
            self.perturbation_array,
            self.tolerance,
        )
        for hash_index in range(self.perturbations):
            distinct, first_index = np.unique(hashes[hash_index], return_index=True)
            new = ~self._hashes_seen(hash_index, distinct)
            self.unique_counts[start + first_index[new]] += 1
            self._pending_hashes[hash_index].update(distinct[new].tolist())
            self._pending_total += int(new.sum())
        self._rows_in_tables = max(self._rows_in_tables, start + len(rows))
        self._after_adding_hashes(range(self.perturbations))

    def add_vectors(self, vectors: np.ndarray) -> None:
        """Append many vectors to the dataset and the hash tables at once.

        Gives the same result as appending them one at a time with
        ``add_input_vector_to_dda()``, much faster.  For an out-of-core dataset
        the store must be appendable (HDF5).
        """
        vectors = np.atleast_2d(np.asarray(vectors))
        self.pre_dda_processing()
        if vectors.shape[1] != self.store.dim:
            raise ValueError("Added vectors have the wrong length for this dataset.")
        self._require_hash_tables("Adding vectors")
        if self.is_out_of_core and len(self.store) > self.vector_count:
            # rows already in the store but not in the tables (an earlier
            # add_vectors failed part-way): hash them first so rows stay aligned
            for row, rows in self.store.iter_chunks(
                self._add_chunk_rows(), self.vector_count, len(self.store)
            ):
                self._add_rows_to_tables(rows, row)
                self.vector_count = row + len(rows)
        start = self.vector_count
        if self.is_out_of_core:
            self.store.append(vectors)
        else:
            if start + len(vectors) > self._dataset_array.shape[0]:
                raise ValueError(
                    "New dataset array size exceeds maximum allowed size; raise "
                    "max_vector_array_size or use an appendable out-of-core dataset."
                )
            self._dataset_array[start : start + len(vectors)] = vectors
        rows_per_chunk = self._add_chunk_rows()
        for offset in range(0, len(vectors), rows_per_chunk):
            chunk = vectors[offset : offset + rows_per_chunk]
            self._add_rows_to_tables(chunk, start + offset)
            # advance per chunk: if a later chunk fails, vector_count still
            # matches the rows in the tables
            self.vector_count = start + offset + len(chunk)

    def flush(self) -> None:
        """Merge the hashes of appended vectors into the tables and, with an
        ``index_dir``, save them, so the index on disk covers every vector.

        Without ``flush()``, an index still recovers vectors appended to an
        appendable store (HDF5) when it is reopened, by hashing them again.
        """
        if self.sorted_hashes is None or self._dirty:
            return  # nothing built yet (or a rebuild is due, which starts afresh)
        if self.index_dir is None:
            for hash_index in range(self.perturbations):
                if self._pending_hashes[hash_index]:
                    self._merge_pending_hashes(hash_index)
            return
        self._save_index()

    def _save_index(self) -> None:
        """Write the pending hashes into new table files and commit them with the counts."""

        # Commit exactly the rows whose hashes are in the tables: a flush can
        # happen part-way through adding vectors, before vector_count moves on.
        rows = self._rows_in_tables
        store = (
            self.store
            if self.is_out_of_core
            else InMemoryStore(self._dataset_array[:rows])
        )
        changed = [p for p in range(self.perturbations) if self._pending_hashes[p]]
        for hash_index in changed:
            self.index.write_table(hash_index, self._merged_table(hash_index))
        self.index.commit(self.unique_counts[:rows], store)
        for hash_index in changed:
            self.sorted_hashes[hash_index] = self.index.load_table(hash_index)
            self._pending_hashes[hash_index].clear()
        self._pending_total = 0
        logger.info("Saved the hash index %s (%d vectors).", self.index_dir, rows)

    def duplicate_check(self) -> bool:
        """duplicate check against dataset using multi-hashing approach.

        Returns:
            bool: True if a duplicate is detected, False otherwise.
        """
        self.pre_dda_processing()
        self._require_hash_tables("duplicate_check()")
        clash_vector = self._clash_vector(self.create_hash_vector())
        duplicate_vote_array = np.sum(clash_vector) / self.perturbations

        duplicate_structure = duplicate_vote_array >= self.acceptance_threshold

        return bool(duplicate_structure)

    def get_uniqueness_score(self) -> float:
        self.pre_dda_processing()
        self._require_hash_tables("get_uniqueness_score()")
        clash_vector = self._clash_vector(self.create_hash_vector())
        uniqueness_score = 1 - np.sum(clash_vector) / self.perturbations

        return uniqueness_score

    def get_dataset_unique_structures(self) -> int:
        """Mark as unique each vector that is the earliest in its hash bucket in
        at least ``acceptance_threshold`` of the perturbations.

        Returns:
            int: The number of unique structures in the dataset.
        """
        self.pre_dda_processing()
        counts = self.unique_counts[: self.vector_count]
        self.unique_vector_indices = (
            counts / self.perturbations >= self.acceptance_threshold
        )

        return int(np.sum(self.unique_vector_indices))

    def _perturbation_array_stale(self) -> bool:
        """Whether the perturbation array must be (re)generated from ``seed``."""
        if self.perturbation_array.shape[0] != self.perturbations:
            return True
        generated_from = getattr(self, "_perturbation_seed", None)
        return generated_from is not None and generated_from != self.seed

    def _hashed_with_other_perturbations(self) -> bool:
        hashed = getattr(self, "_hashed_perturbation_array", None)
        return hashed is None or not np.array_equal(hashed, self.perturbation_array)

    def pre_dda_processing(self) -> None:
        if (
            self.tolerance != getattr(self, "_hash_tolerance", None)
            or self._perturbation_array_stale()
            or self._hashed_with_other_perturbations()
            or self.unique_counts.shape[0] < self.vector_count
            or (
                self.sorted_hashes is not None
                and len(self.sorted_hashes) != self.perturbations
            )
        ):
            self._dirty = True
        super().pre_dda_processing()

    def _rebuild_auxiliary_structures(self) -> None:
        if self._perturbation_array_stale():
            self.set_perturbation_array()
        self.compute_hash_tables()
        self._hash_tolerance = self.tolerance
        self._hashed_perturbation_array = np.array(self.perturbation_array, copy=True)

    def _append_vector_to_structures(self) -> None:
        """Add the input vector to the dataset (array or appendable store) and update the hash tables accordingly."""
        self._require_hash_tables("Adding vectors")
        if self.is_out_of_core:
            self.store.append(self.input_vector.reshape(1, -1))
        else:
            if self.vector_count >= self._dataset_array.shape[0]:
                raise ValueError(
                    "New dataset array size exceeds maximum allowed size; raise "
                    "max_vector_array_size or use an appendable out-of-core dataset."
                )
            self._dataset_array[self.vector_count] = self.input_vector
        self.add_input_vector_hashes(self.input_vector)
        self.vector_count += 1
