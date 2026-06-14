# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`deduplicate_lib` is a Python package implementing deduplication algorithms for numpy vector arrays, plus algorithms for automatically tuning the "tolerance" (distance threshold) used to decide whether two vectors are duplicates.

## Common Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run all tests
pytest

# Run a specific test file or test
pytest tests/unit/core/test_duplicate_detection_algorithm.py
pytest tests/unit/core/test_duplicate_detection_algorithm.py::test_name -v

# Run with coverage (CI uses this)
pytest --cov=src --cov-report=xml

# Lint
ruff check .
ruff format .
```

## Architecture

The library is organized around a **plugin/factory architecture** with two abstract base classes:

- `DuplicateDetectionAlgorithm` (`src/deduplicate_lib/core/duplicate_detection_algorithm.py`) — abstract base for algorithms that detect duplicate vectors. Owns the dataset array, distance matrix, tolerance, and distance metric.
- `ToleranceCalculator` (`src/deduplicate_lib/core/tolerance_calculator.py`) — abstract base for algorithms that find a good tolerance value for a given `DuplicateDetectionAlgorithm` + dataset.

Concrete implementations live under `src/deduplicate_lib/plugins/` and register themselves via the `@register_plugin(kind, name)` decorator (`src/deduplicate_lib/core/plugin_registry.py`). Plugins are instantiated via `create_plugin(kind, name, **kwargs)` rather than imported directly, so the registry must have already imported the module containing the plugin for it to be available.

### Duplicate detection algorithms (`plugins/duplicate_detection_algorithms/`)

- **`distance_matrix.py`** (`DistanceMatrix`, kind=`duplicate_detection_algorithm`, name=`distance_matrix`) — computes a full pairwise distance matrix and flags duplicates below `tolerance`. Simple and accurate but expensive (O(n²)).
- **`multi_hashing.py`** (`MultiHashing`, name=`multi_hashing`) — smears each vector with random perturbations, rounds to `tolerance`, and hashes. Duplicates are decided by the proportion of hash clashes across perturbations vs. `acceptance_threshold` (derived from `sigma_accepatnce_threshold`, 1-4). Fast, approximate. Only supports `hamming` distance.

Both subclass `DuplicateDetectionAlgorithm` and implement `duplicate_check()` and `get_dataset_unique_structures()`.

### Tolerance calculators (`plugins/tolerance_calculators/`)

Both rely on `ToleranceCalculator.binary_search_tolerance()`, which repeatedly swaps in a `tolerance_dataset_array`, sets a trial `tolerance` via the `temp_attr` context manager, and calls `get_dataset_unique_structures()` to binary-search for a tolerance that yields a target number of unique vectors.

- **`perturbed_dataset_reclustering.py`** (`PerturbedDatasetReclustering`, name=`perturbed_dataset_reclustering`) — finds the tolerance(s) that reproduce a target number of unique vectors in a perturbed copy of the dataset (`average`/`loose`/`tight` modes).
- **`natural_tolerance_plateau_probe.py`** (`NaturalTolerancePlateauProbe`, name=`natural_tolerance_plateau_probe`) — binary-searches the "all same"/"all different" tolerance bounds, then probes the range between them and finds a plateau in the unique-structure-count curve to pick a stable tolerance.

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
