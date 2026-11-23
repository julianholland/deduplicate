# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`deduplicate_lib` is a Python package implementing deduplication algorithms for numpy vector arrays, plus algorithms for automatically tuning the "tolerance" (distance threshold) used to decide whether two vectors are duplicates.

## Common Commands

```bash
# Create .venv and install the package plus the dev dependency group (from uv.lock)
uv sync

# Run all tests
uv run pytest

# Run a specific test file or test
uv run pytest tests/unit/core/test_duplicate_detection_algorithm.py
uv run pytest tests/unit/core/test_duplicate_detection_algorithm.py::test_name -v

# Run with coverage (CI uses this)
uv run pytest --cov=src --cov-report=xml

# Lint
uv run ruff check .
uv run ruff format .

# After changing dependencies in pyproject.toml (CI runs `uv sync --locked`)
uv lock

# Build sdist + wheel
uv build
```

Packaging uses hatchling + hatch-vcs: the version comes from git tags (`vMAJOR.MINOR.PATCH`; releases up to `0.1.2` used bare tags) and is written to the untracked `src/deduplicate_lib/version.py` at build/install time — there is no version number to edit. Releases go through `/publish`. Dev/test dependencies are PEP 735 dependency groups (`dev` includes `test`), not extras; the Sphinx deps remain a `docs` extra so Read the Docs can install them. Ruff uses ALomancy's rule set (`E,W,F,I,B,C4,UP,ARG,SIM,RUF`), and the formatter skips Markdown.

## Architecture

The library is organized around a **plugin/factory architecture** with two abstract base classes:

- `DuplicateDetectionAlgorithm` (`src/deduplicate_lib/core/duplicate_detection_algorithm.py`) — abstract base for algorithms that detect duplicate vectors. Owns the dataset array, distance matrix, tolerance, and distance metric.
- `ToleranceCalculator` (`src/deduplicate_lib/core/tolerance_calculator.py`) — abstract base for algorithms that find a good tolerance value for a given `DuplicateDetectionAlgorithm` + dataset.

Concrete implementations live under `src/deduplicate_lib/plugins/` and register themselves via the `@register_plugin(kind, name)` decorator (`src/deduplicate_lib/core/plugin_registry.py`). Plugins are instantiated via `create_plugin(kind, name, **kwargs)` rather than imported directly, so the registry must have already imported the module containing the plugin for it to be available.

### Duplicate detection algorithms (`plugins/duplicate_detection_algorithms/`)

- **`distance_matrix.py`** (`DistanceMatrix`, kind=`duplicate_detection_algorithm`, name=`distance_matrix`) — computes a full pairwise distance matrix; in dataset order, a vector is kept unless it is within `tolerance` of a vector already kept, so each group of near-duplicates keeps its earliest member. Simple and accurate but expensive (O(n²)).
- **`multi_hashing.py`** (`MultiHashing`, name=`multi_hashing`) — smears each vector with random perturbations, rounds to `tolerance`, and hashes with the deterministic numba kernel `fast_quantise_and_hash` (never Python's salted `hash()`). Hash tables are flat: `sorted_hashes` (one sorted `uint64` array of distinct hashes per perturbation, plus `_pending_hashes` sets for appended vectors, merged periodically) and `unique_counts` (per vector, the number of perturbations in which it is the earliest of its hash bucket, built with `np.unique(..., return_index=True)`). A vector is unique if `unique_counts / perturbations >= acceptance_threshold` (derived from `sigma_accepatnce_threshold`, 1-4); `duplicate_check()` uses the fraction of the input's hashes found in the tables. Fast, approximate. Only supports `hamming` distance. `tests/unit/plugins/duplicate_detection_algoirthms/reference_multi_hashing.py` keeps the old dict-based algorithm as a test oracle.

Both subclass `DuplicateDetectionAlgorithm` and implement `duplicate_check()` and `get_dataset_unique_structures()`. Contract for the latter: every group of near-duplicates keeps exactly one vector (its earliest); never flag a whole group.

### Tolerance calculators (`plugins/tolerance_calculators/`)

Both draw a sample of `sample_size` rows from `dda.store` (`draw_sample`: `blocks` = `VectorStore.sample_blocks`, `random`, `head`), build `tuning_array` (hook `_build_tuning_array`) and tune on `dda.tuning_copy(tuning_array)` — a new in-memory DDA of the same class/settings (`_init_kwargs()`, extended by `MultiHashing`). They never modify the DDA passed in (no dataset swapping). `binary_search_tolerance(tuning_dda, target, find_largest)` bisects `tuning_dda.tolerance` for a target unique count.

- **`perturbed_dataset_reclustering.py`** (`PerturbedDatasetReclustering`, name=`perturbed_dataset_reclustering`) — follows each sampled vector with `perturbations_per_vector - 1` noisy copies and finds the tolerance(s) at which they collapse onto the originals (target defaults to the number sampled; `average`/`loose`/`tight`). Samples `random` by default, since it assumes sampled rows are distinct.
- **`natural_tolerance_plateau_probe.py`** (`NaturalTolerancePlateauProbe`, name=`natural_tolerance_plateau_probe`) — bisects the "all different"/"all same" tolerances, sweeps log-spaced tolerances `SWEEP_MARGIN` beyond both, and finds plateaus (flat regions of count vs log tolerance; equal-count runs merged) as `Plateau` objects; the end ones are flagged `trivial`. `find_natural_plateaus()` lists them, `calculate_tolerance("longest"|"tightest"|"loosest")` returns a non-trivial one's geometric midpoint. Samples `blocks` by default, because a uniform sample of a large dataset contains almost no near-duplicate pairs.

### Out-of-core datasets

`core/vector_store.py`: `VectorStore` (shape/dtype/`read(start, stop)`, plus `iter_chunks`, `take`, `sample`, `sample_blocks`, `append`, `write_subset`) with `InMemoryStore`, `NpyStore`, `HDF5Store` (h5py imported lazily, `hdf5` extra), `CallableStore`; `open_store()` dispatches on type/extension. A DDA given `dataset=` (path or store) sets `_store` and `vector_count = len(store)`; `is_out_of_core` is then true, `dataset_array`/`get_filled_dataset_array()` raise, and `preinitialize_dataset_array()` is a no-op. `dda.store` always works (an `InMemoryStore` view over the filled rows for in-memory data), so algorithms should read through `self.store.iter_chunks(self.chunk_rows())`. `core/resources.py` turns `memory_limit` into chunk rows (¼) and perturbation batches (½); `MultiHashing` keeps `sorted_hashes` only if they fit in ¼ (else `sorted_hashes = None` and `_require_hash_tables` raises). `DistanceMatrix._check_fits_in_memory` refuses stores and oversize matrices. `set_dataset_array()` switches a DDA back to in-memory.

`core/hash_index.py` (`index_dir`): `metadata.json` is the atomic commit point naming generation-numbered `p<PPPP>.g<N>.npy` / `counts.g<N>.npy` files; files are only ever written under new names (staged) and become current at `commit()`, unreferenced ones are deleted on open/commit. Never commit counts for rows whose hashes aren't all in the committed tables (or vice versa): `MultiHashing._rows_in_tables` tracks the rows covered, and `_save_index()` commits exactly those. Public `flush()` is a no-op while `_dirty`; code running inside a rebuild must call `_save_index()`. Pending hashes are never in the tables, so merges use sorted insertion (`_merged_table`), not `union1d`. With an index, a save writes pending hashes as a new delta segment per perturbation (`delta_files` in metadata, format 3; `_delta_hashes` in memory) and compacts geometrically (`_stage_pending_segment`, `COMPACTION_RATIO`); lookups go through `_segments()`. Segments of one perturbation never share a hash. Index segments are `SortedSegment`s (`core/sorted_segment.py`): a memory-mapped table plus an in-memory fence (first hash of each file page, saved as a `.fence.npy` sidecar next to each segment and kept/cleaned with it), so a cold lookup reads one page; `_clash_vector` prefetches a vector's pages with `madvise(MADV_WILLNEED)` before looking them up. In-memory MH keeps plain arrays. Pending hashes are `PendingHashes` (`core/pending_hashes.py`: sorted uint64 arrays merged geometrically, plus a small set for single additions), never Python sets of all of them. The counts file (`counts.g<N>.bin`, raw uint32) is append-only across saves: `HashIndex.commit(..., unchanged_rows=...)` appends when every committed count is unchanged (cutting off bytes past `n_vectors` left by an interrupted commit), else writes a new file (the build). Tests disable fsync via `tests/unit/plugins/duplicate_detection_algoirthms/conftest.py` (it's slow on btrfs).

### Pre-allocation pattern

Dataset arrays and distance matrices are pre-allocated to `max_vector_array_size` (default 10000) rather than grown dynamically. `_dataset_array` is the writable backing array; the public `dataset_array` property returns a read-only view. Always mutate via `set_dataset_array()` / `_set_dataset_array_internal()`, never assign directly. `preinitialize_dataset_array()` and `pre_dda_processing()` (overridden per-plugin) must be called before duplicate checks that depend on precomputed structures (distance matrix / hash dictionary).

### Numba

Hot loops (`fast_compute_distance_matrix`, `fast_get_new_distance_matrix_column`, distance functions, `fast_round_and_perturb`) are `@njit`-compiled and defined at module level (not as methods) since they must be jit-compatible. Lines inside these are marked `# pragma: no cover` since numba-compiled code doesn't show up in coverage reports despite being tested.

## Tests

- `tests/unit/` mirrors `src/deduplicate_lib/` structure (`core/`, `plugins/duplicate_detection_algoirthms/` [sic], `plugins/tolerance_calculators/`).
- `tests/integration/` covers end-to-end flows including the example notebook (`test_example_notebook.py`).
- `tests/conftest.py` provides shared fixtures including `DummyDDA`/`DummyToleranceCalculator` minimal implementations and pre-built `distance_matrix_dda`/`multi_hashing_dda` fixtures.

## Agent Delegation Rules
- Use Haiku subagents for: file scanning, search, bulk edits, boilerplate
- Use Sonnet subagents for: code review, test writing, refactoring
- Reserve Opus (main agent) for: architecture decisions, complex debugging, multi-step reasoning
- Always spawn subagents for parallelizable tasks
