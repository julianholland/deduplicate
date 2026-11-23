# Changelog

All notable changes to this project are documented in this file.

## [1.1.0] - 2026-11-23

### Changed
- `MultiHashing(index_dir=...)` saves appended hashes incrementally. A save used to
  rewrite every perturbation's whole table (tens of GB for tens of millions of
  vectors), so the cost of appending grew with the size of the index. Now each save
  writes the new hashes as a small sorted **delta segment** per perturbation, and
  segments of similar size are merged (a new segment absorbs older ones up to
  `COMPACTION_RATIO` = 2 times its size, and becomes the new base table once it
  reaches half of it). A table holds O(log n) segments, each hash is rewritten
  O(log n) times, and the amortised cost of an append no longer grows with the index.
  Lookups search the base table and every delta.
- Index format 3: `metadata.json` gains `delta_files` (per perturbation, oldest
  first). Format 2 indexes still open and are upgraded on their next commit; older
  deduplicate_lib versions refuse format 3.
- Index lookups read one page of a table that does not fit in the page cache. Each
  segment file has a small **fence** sidecar (`<segment>.fence.npy`, the first hash
  of every 4 KiB page, ~1/512 of the table) kept in memory as a `SortedSegment`
  (`core/sorted_segment.py`), so a lookup searches the fence and then a single page
  instead of binary-searching the memory-mapped table (~log2 n cold pages). A single
  vector's lookups (`duplicate_check()`, `add_input_vector_to_dda()`) first ask the
  kernel for all their pages at once (`madvise(MADV_WILLNEED)`), so the disk reads
  them concurrently. Missing sidecars are recomputed when a segment is loaded.
- Pending hashes (added since the last merge / index save) are kept in sorted
  `uint64` arrays (`core/pending_hashes.py`, `PendingHashes`: ~8 bytes per hash)
  instead of Python sets (~60-100 bytes per hash), so many more vectors can be added
  between saves; single additions go through a small set that spills into an array.
  An index now saves when pending hashes pass `memory_limit / 4 / 16` (was `/ 100`),
  i.e. less often for the same `memory_limit`.
- Index saves append the new vectors' unique counts to the counts file (now raw
  `counts.g<N>.bin`, only the first `n_vectors` entries committed) instead of
  rewriting all of them, so a save no longer costs O(dataset size). `.npy` counts
  files from earlier versions still open; the build still writes a new file.

### Documentation
- `docs/large_datasets.rst`: what an index save writes (only new segments and
  counts), how lookups behave once the index outgrows memory (one page per lookup;
  prefer large `add_vectors()` batches), and the disk space for fences and merges.

## [1.0.0] - 2026-11-17

### Changed
- `MultiHashing` was rewritten around flat `uint64` hash tables: per perturbation, a
  sorted array of the dataset's distinct hashes (`sorted_hashes`), plus a per-vector
  count of the perturbations in which it is the earliest vector of its hash bucket
  (`unique_counts`). `get_dataset_unique_structures()` is now O(P·n log n) instead of
  O(P·n²) and uses a fraction of the memory, which is the first step towards
  out-of-core deduplication of very large datasets. Results are unchanged: the unique
  masks, `duplicate_check()` and `get_uniqueness_score()` are tested against the old
  implementation.
- Hashes come from a new numba kernel, `fast_quantise_and_hash`, instead of Python's
  `hash()`, so they are the same in every process (Python salts `hash()` per process)
  and can be persisted.
- Progress output of the tolerance calculators goes through `logging`
  (`deduplicate_lib.*` loggers) instead of `print`; leftover debug prints were removed.

### Added
- `DuplicateDetectionAlgorithm.get_unique_vector_mask()` returns the boolean mask of
  unique vectors. `get_unique_vector_indices()` still returns integer indices.
- Out-of-core datasets: pass `dataset=` (a `.npy`/`.h5`/`.hdf5` path or a
  `VectorStore`) instead of `dataset_array=`, and `MultiHashing` reads the vectors in
  chunks instead of loading them. New `deduplicate_lib.core.vector_store` module:
  `InMemoryStore`, `NpyStore` (memory-mapped), `HDF5Store` (appendable),
  `CallableStore` (adapter for databases and other sources), `open_store()`.
- `memory_limit` (bytes, `"8GB"`, or `"auto"` = half the available memory) sizes the
  row chunks and perturbation batches. `MultiHashing` keeps the per-perturbation hash
  tables only if they fit in a quarter of it; otherwise it keeps just the unique
  counts, and `duplicate_check()`, `get_uniqueness_score()` and appends raise
  `MemoryError`.
- `n_threads`: `fast_quantise_and_hash` is now parallel (numba `prange`).
- `progress=True` shows tqdm progress bars.
- `deduplicate(out=...)` streams the unique vectors to a `.npy`/HDF5 file and returns
  it as a store. Out-of-core datasets write `<input>_unique.<ext>` by default; for
  in-memory datasets `deduplicate()` still returns an array.
- `MultiHashing(index_dir=...)` keeps the hash tables on disk (memory-mapped), so
  `duplicate_check()` and appends work at any dataset size. The index is reused when
  the dataset and hashing parameters match (`IndexMismatchError` names what differs),
  resumes an interrupted build (it commits after every perturbation batch), and
  recovers vectors appended to an HDF5 dataset without a `flush()`. Commits are
  atomic (`metadata.json` names the current generation-numbered files) and fsynced;
  see `deduplicate_lib.core.hash_index`.
- `MultiHashing.add_vectors()` appends many vectors at once (same result as appending
  one at a time) and `MultiHashing.flush()` saves appended vectors to the index.
- `VectorStore.iter_chunks()` takes an optional row range.
- Optional extras `hdf5` (h5py) and `progress` (tqdm).
- Docs: a "Large datasets" page.

### Changed (DistanceMatrix)
- `DistanceMatrix` refuses out-of-core datasets (`ValueError`) and raises
  `MemoryError` when its `max_vector_array_size`² matrix would exceed `memory_limit`,
  pointing to `MultiHashing`. Pass a larger `memory_limit` to override.

### Changed (tolerance calculators, breaking)
- Both tolerance calculators tune on a sample of the dataset read through
  `dda.store`, so they work on out-of-core datasets, and run every search on a
  separate in-memory copy of the DDA (`DuplicateDetectionAlgorithm.tuning_copy()`):
  the DDA passed in (its dataset, store, `index_dir` and hash tables) is no longer
  modified. New arguments `sample_size` (rows tuned on), `sampling` (`"blocks"`,
  `"random"`, `"head"`), `block_rows` and `seed`; all arguments after the DDA are
  keyword-only.
- `PerturbedDatasetReclustering` builds its noisy copies from the sample (default
  `perturbations_per_vector=5`, `sample_size=500`, `sampling="random"`); the default
  target is the number of sampled vectors. An invalid
  `target_unique_vectors_threshold` raises in `__init__`.
- `NaturalTolerancePlateauProbe` sweeps log-spaced tolerances from below the
  "all different" to above the "all same" tolerance (it used a linear sweep between
  10%/90%-unique bounds, which missed fine clustering scales), detects plateaus
  against log tolerance, merges neighbouring plateaus with the same count, and returns
  a plateau's geometric midpoint. New `find_natural_plateaus()` returns every plateau
  as a `Plateau` (start/end/midpoint tolerance, unique count, steps, `trivial`);
  `calculate_tolerance(condition=...)` takes `"longest"` (default), `"tightest"` or
  `"loosest"` and never picks the trivial all-unique/all-same plateaus. Default
  `sampling="blocks"`. The gradient window (`datapoints_to_calculate_gradient`) now
  defaults to 10% of `probe_steps` (it was 3 steps), chosen from a study across
  synthetic datasets, sizes, dimensions and both DDAs; see "Choosing the gradient
  window" in the docs and `docs/scripts/ntpp_window_study.py`. Fixes plateau detection mis-pairing starts and ends when the
  sweep begins inside a plateau.
- `memory_limit="auto"` uses `MemAvailable` on Linux; it used the free physical
  pages, which leave out reclaimable page cache and were often a small fraction of
  the usable memory.

### Added (tolerance calculators)
- `VectorStore.sample_blocks()` and `deduplicate_lib.core.tolerance_calculator.draw_sample()`.

### Fixed
- `deduplicate()` recomputes the unique mask first; it returned rows chosen by a stale
  mask after a `tolerance` change or a same-size `set_dataset_array()`.
- `MultiHashing` keeps a user-supplied `pertrubation_array`; it was replaced by one
  generated from `seed` on the first build. A seed-generated array is regenerated when
  `seed` changes, and assigning a new array triggers a rebuild.
- `deduplicate(out=...)` refuses to write over its own input file.
- `DistanceMatrix.set_dataset_store()` with an on-disk dataset raises before dropping the
  current data, leaving the DDA usable.
- `DistanceMatrix` only refuses (`MemoryError`) a distance matrix larger than an
  explicit `memory_limit`; with the default `"auto"` (which depends on the memory free
  at that moment) it warns (`ResourceWarning`).
- `MultiHashing.add_vectors()` keeps `vector_count` in step with the hash tables if it
  fails part-way, and hashes rows left in an out-of-core store by such a failure before
  appending more.
- `VectorStore.take()`/`sample()` read only the requested rows (memory-mapped fancy
  indexing for `.npy`, one point selection for HDF5) instead of a 65536-row chunk per row.
- The hash index fingerprints every indexed row (sha256 per block of 2^20 rows,
  extended incrementally as the index grows) instead of the first and last 1024 rows,
  so an index is no longer reused after rows in the middle of the dataset change.
  Opening an index reads the indexed rows once. Index format version 2.

### Known issues
- Peak memory while building hash tables can exceed `memory_limit` by roughly
  1.2-1.5x (`np.unique`'s working copies are not budgeted).
- Many small `MultiHashing.add_vectors()` calls in `index_dir` mode slow down with the
  number of pending (unflushed) hashes; call `flush()` periodically or add in larger
  batches.

### Removed (breaking)
- `ToleranceCalculator(tolerance_dataset_array=...)`, `temp_attr`,
  `create_perturbed_dataset()`, `_ensure_perturbed_dataset()`, and the
  `perturbations_per_vector`/`perturbation_scale` arguments of the base class and
  `NaturalTolerancePlateauProbe`; `NaturalTolerancePlateauProbe(probe_buffer_fraction=...)`
  and its `"minimum"`/`"maximum"` conditions (use `"tightest"`/`"loosest"`).
- `MultiHashing.hash_dict` and its helpers (`initialize_hash_vector_dictionary`,
  `compute_hash_vector_dictionary`, `add_input_vector_hashes_to_dictionary`,
  `warn_if_vector_dict_mismatch`) and `fast_round_and_perturb`. Use
  `sorted_hashes`/`unique_counts`, `initialize_hash_tables()`, `compute_hash_tables()`
  and `add_input_vector_hashes()` instead. `create_hash_vector()` now returns `uint64`
  hashes, whose values differ from the old ones.

## [0.1.3] - 2026-10-12

### Changed
- Packaging moved to uv + hatchling: dev/test dependencies are now PEP 735 dependency
  groups (`uv sync` installs them) instead of the `dev` extra, `uv.lock` pins the
  development environment, and CI installs with `uv sync --locked` and tests on
  Python 3.10–3.12.
- Minimum Python version raised from 3.8 to 3.10 (current numba releases already
  require it).
- The version now comes from git tags via hatch-vcs instead of a hand-edited
  `__version__` (which had drifted: the `0.1.2` tag still said `0.1.1`).
  `deduplicate_lib.__version__` still works. New tags use the `vMAJOR.MINOR.PATCH` form.
- Ruff now enforces the same rule set as ALomancy (isort, bugbear, pyupgrade,
  comprehensions, unused arguments, simplify, ruff-specific). Code was updated to
  match: sorted imports, builtin generics, `stacklevel=2` on all warnings, so they
  point at the caller instead of inside the library.
- `DuplicateDetectionAlgorithm.pre_dda_processing()` (and the `MultiHashing`
  override) no longer accept unused `*args, **kwargs`.

### Fixed
- `DistanceMatrix.get_dataset_unique_structures()` now keeps exactly one vector from
  each group of near-duplicates: in dataset order, a vector is unique unless it is within
  `tolerance` of a vector already kept. Before, a vector counted as unique only if
  nothing at all was within `tolerance` (row 0 excepted), so every group that didn't
  contain row 0 was flagged in full and `deduplicate()` dropped it entirely. This matches
  `MultiHashing`, which already keeps the first vector of each group.

### Behaviour change
- Unique counts from `DistanceMatrix` are higher wherever near-duplicate groups exist,
  so the tolerances that `NaturalTolerancePlateauProbe` and
  `PerturbedDatasetReclustering` find with it shift. In the integration tests,
  0.168 → 0.181 (reclustering) and 0.141 → 0.153 (plateau probe). At the old values,
  one point plus a 30-vector cluster deduplicated to the point alone; at the new
  values, to the point plus one cluster member.

---

## [0.1.2] - 2026-09-23

### Fixed
- `NaturalTolerancePlateauProbe`: fixed selection of the minimum/maximum plateau.

---

## [0.1.1] - 2026-09-23

### Added
- `NaturalTolerancePlateauProbe` can select the minimum or maximum plateau.

---

## [0.1.0] - 2026-06-14

### Fixed
- `MultiHashing.duplicate_check()` / `get_uniqueness_score()` no longer return stale results
  when `tolerance`, `seed`, or `perturbations` change after the hash dictionary was first
  built — `pre_dda_processing()` now detects these changes and forces a rebuild.
- `DistanceMatrix.duplicate_check()` no longer triggers an unnecessary full O(N²)
  distance-matrix rebuild (and large preallocation) on a clean dataset; it now only ensures
  the dataset array is preinitialized.
- `set_dataset_array()` is now a no-op (no reallocation, no `_dirty` invalidation) when the
  incoming array is identical to the data already loaded, avoiding redundant rebuilds when
  tolerance calculators temporarily swap datasets and restore the original.
- Removed a redundant double-computation of the hash vector dictionary in
  `MultiHashing.get_dataset_unique_structures()`.
- `DistanceMatrix._rebuild_auxiliary_structures()` no longer redundantly reinitializes the
  distance matrix before recomputing it.

### Changed
- Dropped the unused `input_dataset_array` parameter from `pre_dda_processing()`.
- Extracted `MultiHashing._perturbation_array_stale()` to share the perturbation-array
  staleness check between `pre_dda_processing()` and `_rebuild_auxiliary_structures()`.

---

## [0.0.5] - 2026-04-23

### Changed
- Reinstated dictionary-based storage in multi-hashing implementation (`b3f3fad`).
- Verified full test and coverage status for the dictionary reversion (`b3f3fad`).
- Finalized release/build readiness updates (`ff9ee21`, `02614ed`).

### Merged
- PR #9: Reinstate dictionaries for MH (`aa9aff4`).

---

## [0.0.4] - 2026-04-22

### Added
- Pre-initialisation logic for dataset handling (`60736ff`).

### Changed
- Coverage/test completeness improvements (`74658d4`).

---

## [0.0.3] - 2026-04-16

### Added
- Tight and loose options to PDR (`eca4ab0`).
- `get_uniqueness_score` support in multi-hash flow (`671ddac`).

### Changed
- Acceptance threshold moved to sigma-based threshold (`62d80e6`).
- Round-then-perturb bug fix and hash-array/input reset fixes (`671ddac`).
- Test updates and cleanup (`4d72241`, `47f1271`).
- Version update for v0.0.3 (`ab12fde`).

### Merged
- PR #5: Release v0.0.3 (`97d2102`).

---

## [0.0.2] - 2026-03-30

### Added
- Example usage and deduplicate function introduced (`7ee94df`).
- Additional tests (`85e666a`).
- SciPy dependency (`9c1cfc4`).

### Changed
- Release prep for v0.0.2 (`2239439`).

---

## [0.0.1] - 2026-03-26

### Added
- Initial CI and test setup (`b0b5477`, `bb03869`).
- Numba acceleration for DDAs (`cf10a8e`, `fa93d56`).
- Codecov integration and badge (`033d99f`, `e13a1ba`).
- License and core tolerance calculator tests (`3edf959`).

### Changed
- README and packaging updates (`851b609`, `8108a50`).

---

## [0.0.0] - 2026-03-17

### Added
- Initial project commit (`03a02d8`).