"""Base class for tolerance calculators.

A tolerance calculator tunes ``tolerance`` on a small sample of the DDA's
dataset.  The sample is read through ``dda.store``, so in-memory and
out-of-core datasets are handled the same way, and every search runs on a
detached in-memory copy of the DDA (:meth:`DuplicateDetectionAlgorithm.tuning_copy`):
the DDA passed in is never modified.
"""

import logging
import warnings
from abc import ABC, abstractmethod

import numpy as np

from deduplicate_lib.core.duplicate_detection_algorithm import (
    DuplicateDetectionAlgorithm,
)
from deduplicate_lib.core.vector_store import VectorStore

logger = logging.getLogger(__name__)

SAMPLING_MODES = ("blocks", "random", "head")


def draw_sample(
    store: VectorStore,
    sample_size: int,
    sampling: str = "blocks",
    block_rows: int = 100,
    seed: int = 803,
) -> np.ndarray:
    """Draw up to ``sample_size`` rows of ``store`` for tolerance tuning.

    Parameters
    ----------
    store : VectorStore
        The dataset.
    sample_size : int
        Number of rows to draw (all rows if the dataset is smaller).
    sampling : str
        ``"blocks"``: contiguous blocks of ``block_rows`` from random places,
        which keeps near-duplicates that sit next to each other in the sample;
        ``"random"``: uniformly random rows; ``"head"``: the first rows.
    block_rows : int
        Block length for ``"blocks"``.
    seed : int
        Random seed for ``"blocks"`` and ``"random"``.
    """
    if sampling not in SAMPLING_MODES:
        raise ValueError(
            f"Invalid sampling {sampling!r}. Must be one of {', '.join(SAMPLING_MODES)}."
        )
    if not isinstance(sample_size, int) or sample_size < 1:
        raise ValueError("sample_size must be a positive integer.")
    if len(store) == 0:
        raise ValueError("Cannot tune a tolerance on an empty dataset.")
    if sampling == "blocks":
        sample = store.sample_blocks(sample_size, block_rows=block_rows, seed=seed)
    elif sampling == "random":
        sample = store.sample(sample_size, seed=seed)
    else:
        sample = store.read(0, min(sample_size, len(store)))
    return np.asarray(sample, dtype=float)


class ToleranceCalculator(ABC):
    """Base class for algorithms that choose a tolerance for a DDA.

    Parameters
    ----------
    duplicate_detection_algorithm_object : DuplicateDetectionAlgorithm
        The DDA to tune.  Its settings (metric, hashing parameters, ...) are
        copied; the DDA itself is never modified.
    sample_size : int, optional
        Rows of the dataset to tune on.  Defaults to ``2000``.
    sampling : str, optional
        How the sample is drawn: ``"blocks"`` (default), ``"random"`` or
        ``"head"``; see :func:`draw_sample`.
    block_rows : int, optional
        Block length for ``sampling="blocks"``.  Defaults to ``100``.
    seed : int, optional
        Random seed for sampling (and any perturbation).  Defaults to ``803``.
    binary_search_steps : int, optional
        Iterations of each binary search.  Defaults to ``20``.

    Attributes
    ----------
    tuning_array : np.ndarray
        The array the last calculation tuned on (set by ``_prepare_tuning``).
    """

    def __init__(
        self,
        duplicate_detection_algorithm_object: DuplicateDetectionAlgorithm,
        *,
        sample_size: int = 2000,
        sampling: str = "blocks",
        block_rows: int = 100,
        seed: int = 803,
        binary_search_steps: int = 20,
    ) -> None:
        if sampling not in SAMPLING_MODES:
            raise ValueError(
                f"Invalid sampling {sampling!r}. Must be one of {', '.join(SAMPLING_MODES)}."
            )
        self.duplicate_detection_algorithm_object = duplicate_detection_algorithm_object
        self.sample_size = sample_size
        self.sampling = sampling
        self.block_rows = block_rows
        self.seed = seed
        self.binary_search_steps = binary_search_steps
        self.tuning_array = np.empty((0, 0))

    def _build_tuning_array(self, sample: np.ndarray) -> np.ndarray:
        """Turn the drawn sample into the array to tune on (identity by default)."""
        return sample

    def _prepare_tuning(self) -> DuplicateDetectionAlgorithm:
        """Draw the sample, build ``tuning_array`` and return a DDA copy over it."""
        dda = self.duplicate_detection_algorithm_object
        sample = draw_sample(
            dda.store,
            self.sample_size,
            sampling=self.sampling,
            block_rows=self.block_rows,
            seed=self.seed,
        )
        self.tuning_array = self._build_tuning_array(sample)
        logger.info(
            "Tuning the tolerance on %d of %d vectors (sampling=%r; %d tuning rows).",
            len(sample),
            dda.vector_count,
            self.sampling,
            len(self.tuning_array),
        )
        return dda.tuning_copy(self.tuning_array)

    def tolerance_upper_bound(self) -> float:
        """A tolerance comfortably above the one that makes all tuning rows the same."""
        ptp = np.ptp(self.tuning_array)
        abs_std = np.mean(np.std(self.tuning_array, axis=0))
        return float((ptp + abs_std) * self.tuning_array.shape[1])

    def binary_search_tolerance(
        self,
        tuning_dda: DuplicateDetectionAlgorithm,
        target_unique_vectors: int,
        find_largest_tolerance_for_target: bool = True,
    ) -> float:
        """Binary-search the tolerance at which ``tuning_dda`` has ``target_unique_vectors``.

        Parameters
        ----------
        tuning_dda : DuplicateDetectionAlgorithm
            The DDA copy from ``_prepare_tuning``; its ``tolerance`` is changed.
        target_unique_vectors : int
            Unique-vector count to hit.
        find_largest_tolerance_for_target : bool
            Among tolerances that hit the target, return the largest (else the smallest).

        Returns
        -------
        float
            The tolerance found; if the target is never hit exactly, the
            tolerance with the closest count (with a warning).
        """
        best_min = 0.0
        # Start above the tolerance at which every vector is the same: a more
        # spread-out dataset needs a larger one.
        best_max = self.tolerance_upper_bound()
        tolerance = best_max

        if self.binary_search_steps <= 0:
            warnings.warn(
                f"Binary search called with {self.binary_search_steps} steps. No binary search performed.\n Returning the upper bound of the search range: {tolerance}.",
                stacklevel=2,
            )
            return tolerance

        exact_list = []
        results_dict = {}
        for _ in range(self.binary_search_steps):
            tuning_dda.tolerance = tolerance
            unique_vectors = tuning_dda.get_dataset_unique_structures()

            results_dict[tolerance] = unique_vectors
            if unique_vectors < target_unique_vectors:  # tighten tolerance
                best_max = tolerance
            elif unique_vectors > target_unique_vectors:  # loosen tolerance
                best_min = tolerance
            else:
                exact_list.append(tolerance)
                if (
                    target_unique_vectors == tuning_dda.vector_count
                    or find_largest_tolerance_for_target
                ):
                    best_min = tolerance
                else:
                    best_max = tolerance
            logger.info("Tolerance: %s, Unique Vectors: %s", tolerance, unique_vectors)
            tolerance = (best_min + best_max) / 2

        if exact_list:
            return (
                max(exact_list)
                if find_largest_tolerance_for_target
                else min(exact_list)
            )

        closest_tol = min(
            results_dict,
            key=lambda tol: abs(results_dict[tol] - target_unique_vectors),
        )
        warnings.warn(
            f"No exact tolerance found for target of {target_unique_vectors} unique vectors during binary search.\n Returning closest tolerance found: {closest_tol} with {results_dict[closest_tol]} unique vectors.",
            stacklevel=2,
        )
        return closest_tol

    @abstractmethod
    def calculate_tolerance(self) -> float:
        """Compute and return a tolerance value for the configured DDA.

        Returns
        -------
        float
            A tolerance value suitable for use as ``dda.tolerance``.
        """
        pass  # pragma: no cover, implemented in child classes
