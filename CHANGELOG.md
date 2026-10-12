# Changelog

All notable changes to this project are documented in this file.

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