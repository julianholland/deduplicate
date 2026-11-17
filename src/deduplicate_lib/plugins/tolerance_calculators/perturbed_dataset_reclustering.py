import numpy as np

from deduplicate_lib.core.duplicate_detection_algorithm import (
    DuplicateDetectionAlgorithm,
)
from deduplicate_lib.core.plugin_registry import register_plugin
from deduplicate_lib.core.tolerance_calculator import ToleranceCalculator

THRESHOLDS = ("average", "loose", "tight")


@register_plugin("tolerance_calculator", "perturbed_dataset_reclustering")
class PerturbedDatasetReclustering(ToleranceCalculator):
    """Tolerance calculator based on perturbed-dataset reclustering.

    Draws a sample of the dataset, adds ``perturbations_per_vector - 1`` noisy
    copies of each sampled vector, then binary-searches for the tolerance at
    which the noisy copies collapse back onto their originals, i.e. at which
    the tuning set has ``target_unique_vectors`` unique vectors.  The returned
    value depends on ``target_unique_vectors_threshold``: ``"loose"`` returns
    the highest such tolerance, ``"tight"`` the lowest, and ``"average"`` their
    mean.

    The sampled vectors are assumed to be distinct from each other, so the
    sample is drawn uniformly at random by default (``sampling="random"``),
    which rarely picks two near-duplicates of a large dataset.

    Parameters
    ----------
    duplicate_detection_algorithm_object : DuplicateDetectionAlgorithm
        The DDA to tune; it is not modified.
    perturbations_per_vector : int, optional
        Rows per sampled vector in the tuning set: the original plus
        ``perturbations_per_vector - 1`` noisy copies.  Defaults to ``5``.
    perturbation_scale : float, optional
        Standard deviation of the Gaussian noise of each copy, i.e. how far
        apart two vectors may be and still count as duplicates.  Defaults to ``0.1``.
    target_unique_vectors : int or None, optional
        Target unique-vector count.  ``None`` (default) uses the number of
        sampled vectors.
    target_unique_vectors_threshold : str, optional
        One of ``"average"`` (default), ``"loose"`` or ``"tight"``.
    sample_size : int, optional
        Vectors to sample.  Defaults to ``500`` (``2500`` tuning rows with the
        default ``perturbations_per_vector``).
    sampling : str, optional
        Defaults to ``"random"``.  See :class:`ToleranceCalculator` for this and
        the other sampling arguments (``block_rows``, ``seed``) and
        ``binary_search_steps``.
    """

    def __init__(
        self,
        duplicate_detection_algorithm_object: DuplicateDetectionAlgorithm,
        *,
        perturbations_per_vector: int = 5,
        perturbation_scale: float = 0.1,
        target_unique_vectors: int | None = None,
        target_unique_vectors_threshold: str = "average",
        sample_size: int = 500,
        sampling: str = "random",
        block_rows: int = 100,
        seed: int = 803,
        binary_search_steps: int = 20,
    ):
        super().__init__(
            duplicate_detection_algorithm_object,
            sample_size=sample_size,
            sampling=sampling,
            block_rows=block_rows,
            seed=seed,
            binary_search_steps=binary_search_steps,
        )
        if (
            not isinstance(perturbations_per_vector, int)
            or perturbations_per_vector < 1
        ):
            raise ValueError("perturbations_per_vector must be a positive integer.")
        if target_unique_vectors_threshold not in THRESHOLDS:
            raise ValueError(
                f"Invalid target_unique_vectors_threshold: {target_unique_vectors_threshold}. Must be 'average', 'loose', or 'tight'."
            )
        self.perturbations_per_vector = perturbations_per_vector
        self.perturbation_scale = perturbation_scale
        self.target_unique_vectors = target_unique_vectors
        self.target_unique_vectors_threshold = target_unique_vectors_threshold

    def __str__(self) -> str:
        return f"PerturbedDatasetReclustering(perturbations_per_vector={self.perturbations_per_vector}, perturbation_scale={self.perturbation_scale}, dda={str(self.duplicate_detection_algorithm_object).split('(')[0]})"

    def _build_tuning_array(self, sample: np.ndarray) -> np.ndarray:
        """Each sampled vector followed by its ``perturbations_per_vector - 1`` noisy copies."""
        self._n_sampled = len(sample)
        ppv = self.perturbations_per_vector
        tuning = np.repeat(sample, ppv, axis=0)
        noise = np.random.default_rng(self.seed).normal(
            loc=0.0, scale=self.perturbation_scale, size=tuning.shape
        )
        noise[::ppv] = 0.0  # keep the originals
        return tuning + noise

    def calculate_tolerance(self) -> float:
        """Find the lowest and/or highest tolerance at which the perturbed copies
        collapse onto their originals and combine them as configured.

        Returns
        -------
        float
            The calculated tolerance value.
        """
        tuning_dda = self._prepare_tuning()
        target = (
            self._n_sampled
            if self.target_unique_vectors is None
            else self.target_unique_vectors
        )

        if self.target_unique_vectors_threshold == "loose":
            return self.binary_search_tolerance(
                tuning_dda, target, find_largest_tolerance_for_target=True
            )
        low_tolerance = self.binary_search_tolerance(
            tuning_dda, target, find_largest_tolerance_for_target=False
        )
        if self.target_unique_vectors_threshold == "tight":
            return low_tolerance
        high_tolerance = self.binary_search_tolerance(
            tuning_dda, target, find_largest_tolerance_for_target=True
        )
        return (low_tolerance + high_tolerance) / 2
