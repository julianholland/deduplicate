# Mission Statement — deduplicate_lib

## Why this exists

deduplicate_lib is a **modular library of algorithms that distinguish same-length vectors**, plus tools to tune those algorithms to a user-defined level of granularity.

## Core modules

### `duplicate_detection_algorithms`
Algorithms that detect duplicates. Every algorithm in this module should take a **tolerance** and an **array of vectors**, and be able to return either the list of unique indices, or the array with duplicates removed.

Currently implemented:
- **Distance Matrix** — computes the pairwise distance between all vectors (simple, accurate, expensive)
- **Multi Hashing** — computes the proportion of hash collisions for a smeared and rounded vector (fast)

Planned: **Locality-Sensitive Hashing**, to add a fast-and-accurate option alongside the existing fast/accurate trade-off pair.

### `tolerance_tuning`
Uses a chosen duplicate-detection algorithm to find an appropriate tolerance automatically, so the user doesn't have to guess. Two approaches:
- **PerturbedDatasetReclustering** — the user defines an acceptable noise level for a vector to still count as a duplicate; the tuner finds the tolerance that matches this.
- **NaturalTolerancePlateauProbe** — sweeps a range of tolerances and looks for the largest natural window where the resulting structure count is unaffected, then places the tolerance inside that window.

## Design principles

- **Extensible**: new duplicate-detection algorithms (e.g. LSH) should be able to plug into the same interface without touching `tolerance_tuning`.
- **Easy to use**: the common case should be a single call, e.g.:
  ```python
  b = deduplicate(a, tolerance=0.1, algorithm="multi-hash")
  ```
  returning `a` with duplicates removed, using the chosen algorithm and tolerance.

## Current known gaps (see `README.md` to-do list)

- Benchmark suite for time and robustness across algorithms is not yet implemented
- Locality-Sensitive Hashing is not yet added as a third detection algorithm
