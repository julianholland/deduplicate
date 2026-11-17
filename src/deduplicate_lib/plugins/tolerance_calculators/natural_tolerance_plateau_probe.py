import logging
import warnings
from dataclasses import dataclass

import numpy as np

from deduplicate_lib.core.duplicate_detection_algorithm import (
    DuplicateDetectionAlgorithm,
)
from deduplicate_lib.core.plugin_registry import register_plugin
from deduplicate_lib.core.tolerance_calculator import ToleranceCalculator

logger = logging.getLogger(__name__)

CONDITIONS = ("longest", "tightest", "loosest")
# The sweep runs this factor below the "all different" and above the "all same" tolerance.
SWEEP_MARGIN = 2.0
# Default gradient window, as a share of the probed tolerances.  Noise in the
# unique count spans a fixed share of the sweep whatever probe_steps is; 10%
# suppressed most false plateaus without losing narrow genuine ones across
# synthetic datasets, sizes, dimensions and both DDAs
# (docs/scripts/ntpp_window_study.py).
DEFAULT_WINDOW_FRACTION = 0.10


@dataclass(frozen=True)
class Plateau:
    """A flat stretch of the unique-count-versus-tolerance curve.

    Attributes
    ----------
    start_tolerance, end_tolerance : float
        Tolerance range of the plateau.
    tolerance : float
        Its geometric midpoint, the tolerance a calculator returns for it.
    unique_vectors : int
        Median unique-vector count on the plateau (in the tuning sample).
    steps : int
        Number of probed tolerances it spans.
    trivial : bool
        ``True`` for a plateau on which every sampled vector is unique (the
        bottom of the sweep) or all are the same (the top).
    """

    start_tolerance: float
    end_tolerance: float
    tolerance: float
    unique_vectors: int
    steps: int
    trivial: bool


@register_plugin("tolerance_calculator", "natural_tolerance_plateau_probe")
class NaturalTolerancePlateauProbe(ToleranceCalculator):
    """Tolerance calculator that finds stable tolerances directly from the data.

    No perturbed copies are made.  On a sample of the dataset, the algorithm:

    1. Binary-searches the "all different" tolerance (the largest at which no
       sampled vectors merge) and the "all same" tolerance (the smallest at
       which they all collapse into one).
    2. Sweeps ``probe_steps`` log-spaced tolerances from a factor
       ``SWEEP_MARGIN`` below the first to a factor above the second (so it
       covers clustering scales spanning orders of magnitude) and records the
       unique-vector count at each.
    3. Finds plateaus: runs where the gradient of the count against
       ``log(tolerance)``, over a window of ``datapoints_to_calculate_gradient``
       steps (default: 10% of the sweep), is below ``plateau_threshold`` times the average gradient of the
       sweep.  Neighbouring runs with the same count are merged.

    Each plateau is a natural scale at which the data clusters.  The plateau
    at the bottom of the sweep (every vector unique) and at the top (all the
    same) are trivial and never chosen.  ``find_natural_plateaus()`` returns
    them all; ``calculate_tolerance()`` returns the geometric midpoint of the longest,
    tightest or loosest non-trivial one.

    The sample is drawn in contiguous blocks by default (``sampling="blocks"``),
    which keeps near-duplicates that sit next to each other in the dataset
    together; a uniform random sample of a large dataset contains few of them.

    Parameters
    ----------
    duplicate_detection_algorithm_object : DuplicateDetectionAlgorithm
        The DDA to tune; it is not modified.
    probe_steps : int, optional
        Number of tolerances in the sweep.  Defaults to ``100``.
    datapoints_to_calculate_gradient : int or None, optional
        Window (in probe steps) for the local gradient; greater than 1.  A
        plateau must be longer than the window to be found, and a window much
        shorter than the noise in the curve finds false plateaus.  ``None``
        (default) uses ``DEFAULT_WINDOW_FRACTION`` (10%) of ``probe_steps``.
    plateau_threshold : float, optional
        Gradient below which a window is flat, as a fraction of the average
        gradient across the sweep.  Defaults to ``1e-3``.
    binary_search_steps : int, optional
        Iterations of the "all different" and "all same" searches.
        Defaults to ``50``.
    sample_size, sampling, block_rows, seed : optional
        See :class:`ToleranceCalculator` (``sampling`` defaults to ``"blocks"``).

    Attributes
    ----------
    probe_results : dict
        ``{tolerance: unique count}`` of the last sweep.
    plateau_data : dict
        Probed ``tolerances``, ``unique_counts`` and window ``gradients`` of the
        last sweep, for plotting.
    """

    def __init__(
        self,
        duplicate_detection_algorithm_object: DuplicateDetectionAlgorithm,
        *,
        probe_steps: int = 100,
        datapoints_to_calculate_gradient: int | None = None,
        plateau_threshold: float = 1e-3,
        sample_size: int = 2000,
        sampling: str = "blocks",
        block_rows: int = 100,
        seed: int = 803,
        binary_search_steps: int = 50,
    ) -> None:
        super().__init__(
            duplicate_detection_algorithm_object,
            sample_size=sample_size,
            sampling=sampling,
            block_rows=block_rows,
            seed=seed,
            binary_search_steps=binary_search_steps,
        )
        self.probe_steps = probe_steps
        self.datapoints_to_calculate_gradient = datapoints_to_calculate_gradient
        self.plateau_threshold = plateau_threshold
        self.probe_results: dict = {}
        self.plateau_data: dict = {}

    def __str__(self) -> str:
        return f"NaturalTolerancePlateauProbe(probe_steps={self.probe_steps}, sample_size={self.sample_size}, sampling={self.sampling!r}, dda={str(self.duplicate_detection_algorithm_object).split('(')[0]})"

    def gradient_window(self, n_points: int | None = None) -> int:
        """The gradient window in probe steps: ``datapoints_to_calculate_gradient``,
        or ``DEFAULT_WINDOW_FRACTION`` of ``n_points`` (default ``probe_steps``)."""
        if self.datapoints_to_calculate_gradient is not None:
            return self.datapoints_to_calculate_gradient
        n_points = self.probe_steps if n_points is None else n_points
        return max(2, round(DEFAULT_WINDOW_FRACTION * n_points))

    def tolerance_probe(
        self,
        tuning_dda: DuplicateDetectionAlgorithm,
        lower_tolerance: float,
        upper_tolerance: float,
        tolerance_steps: int,
    ) -> dict:
        """Unique-vector counts of ``tuning_dda`` at log-spaced tolerances."""
        if not 0 < lower_tolerance < upper_tolerance:
            lower_tolerance = upper_tolerance / 1e3
        tolerance_results = {}
        for tol in np.geomspace(lower_tolerance, upper_tolerance, tolerance_steps):
            tuning_dda.tolerance = float(tol)
            tolerance_results[float(tol)] = int(
                tuning_dda.get_dataset_unique_structures()
            )
        return tolerance_results

    def probe(self) -> dict:
        """Run the sweep on a fresh sample and return ``{tolerance: unique count}``."""
        window = self.gradient_window()
        if window <= 1:
            raise ValueError("datapoints_to_calculate_gradient must be greater than 1.")
        if self.probe_steps <= window:
            raise ValueError(
                "Not enough tolerance steps to calculate gradient with the given datapoints_to_calculate_gradient."
            )
        tuning_dda = self._prepare_tuning()
        self.all_same_tolerance = self.binary_search_tolerance(
            tuning_dda, target_unique_vectors=1, find_largest_tolerance_for_target=False
        )
        # The "all different" end: the largest tolerance that still keeps as
        # many vectors as a vanishingly small one (fewer than the sample size
        # if the sample holds exact duplicates).
        tuning_dda.tolerance = self.tolerance_upper_bound() * 1e-9
        self.max_unique_vectors = int(tuning_dda.get_dataset_unique_structures())
        self.all_different_tolerance = self.binary_search_tolerance(
            tuning_dda,
            target_unique_vectors=self.max_unique_vectors,
            find_largest_tolerance_for_target=True,
        )
        # Sweep log-spaced (clustering scales span orders of magnitude), a
        # factor SWEEP_MARGIN past both ends so the trivial plateaus show.
        self.probe_results = self.tolerance_probe(
            tuning_dda,
            self.all_different_tolerance / SWEEP_MARGIN,
            self.all_same_tolerance * SWEEP_MARGIN,
            self.probe_steps,
        )
        return self.probe_results

    def get_plateau_log(self, sorted_tols: list, tolerance_results: dict) -> np.ndarray:
        """Boolean per gradient window: ``True`` where the curve is flat.

        Window ``i`` spans ``sorted_tols[i]`` to
        ``sorted_tols[i + datapoints_to_calculate_gradient]``.  Gradients are
        taken against ``log(tolerance)``, matching the log-spaced sweep.
        """
        window = self.gradient_window(len(sorted_tols))
        if window <= 1:
            raise ValueError("datapoints_to_calculate_gradient must be greater than 1.")
        if len(sorted_tols) <= window:
            raise ValueError(
                "Not enough tolerance steps to calculate gradient with the given datapoints_to_calculate_gradient."
            )

        tols = np.asarray(sorted_tols, dtype=float)
        log_tols = np.log(tols)
        counts = np.asarray(
            [tolerance_results[tol] for tol in sorted_tols], dtype=float
        )
        total_gradient = (counts[-1] - counts[0]) / (log_tols[-1] - log_tols[0])
        gradients = (counts[window:] - counts[:-window]) / (
            log_tols[window:] - log_tols[:-window]
        )
        plateau_log = np.abs(gradients) < abs(self.plateau_threshold * total_gradient)

        self.plateau_data = {
            "tolerances": list(tols),
            "unique_counts": [int(c) for c in counts],
            "gradients": list(gradients),
        }
        return plateau_log

    def find_plateaus(self, tolerance_results: dict) -> list[Plateau]:
        """All plateaus of a sweep, in tolerance order (trivial ones flagged)."""
        sorted_tols = sorted(tolerance_results)
        plateau_log = self.get_plateau_log(sorted_tols, tolerance_results)
        logger.debug("plateau log: %s", plateau_log)
        window = self.gradient_window(len(sorted_tols))
        counts = np.asarray([tolerance_results[tol] for tol in sorted_tols])
        max_unique = getattr(self, "max_unique_vectors", None) or int(counts.max())

        # run-length scan: [first, last] window indices of each run of True
        padded = np.concatenate([[False], plateau_log, [False]]).astype(np.int8)
        edges = np.flatnonzero(np.diff(padded))
        runs = []  # [first point, last point, median count]
        for first, stop in zip(edges[::2], edges[1::2]):
            last_point = stop - 1 + window  # last probe point covered by the run
            unique_vectors = int(np.median(counts[first : last_point + 1]))
            if runs and runs[-1][2] == unique_vectors and first <= runs[-1][1] + window:
                # a short bump (e.g. hashing noise) split one plateau: merge
                runs[-1][1] = last_point
            else:
                runs.append([first, last_point, unique_vectors])

        plateaus = []
        for first, last_point, unique_vectors in runs:
            start_tol = float(sorted_tols[first])
            end_tol = float(sorted_tols[last_point])
            # all unique or all the same, wherever noise lets the run start
            trivial = unique_vectors >= max_unique or unique_vectors <= 1
            plateaus.append(
                Plateau(
                    start_tolerance=start_tol,
                    end_tolerance=end_tol,
                    tolerance=float(np.sqrt(start_tol * end_tol)),
                    unique_vectors=unique_vectors,
                    steps=int(last_point - first + 1),
                    trivial=bool(trivial),
                )
            )
        return plateaus

    def find_natural_plateaus(self, include_trivial: bool = False) -> list[Plateau]:
        """Every plateau of the sweep (running it if needed), in tolerance order.

        Parameters
        ----------
        include_trivial : bool
            Also return the trivial all-unique / all-same plateaus.
        """
        if not self.probe_results:
            self.probe()
        plateaus = self.find_plateaus(self.probe_results)
        return [p for p in plateaus if include_trivial or not p.trivial]

    def calculate_tolerance(self, condition: str = "longest") -> float:
        """Run a fresh sweep and return the midpoint of the chosen natural plateau.

        Parameters
        ----------
        condition : str, optional
            ``"longest"`` (default): the plateau spanning the most probe steps;
            ``"tightest"``: the lowest-tolerance one; ``"loosest"``: the
            highest-tolerance one.  Trivial plateaus are never chosen.

        Returns
        -------
        float
            Midpoint (geometric) of the chosen plateau.  Falls back to the
            geometric mean of the "all different" and "all same" tolerances
            (with a warning) if there is no non-trivial plateau.
        """
        if condition not in CONDITIONS:
            raise ValueError(
                f"Invalid condition '{condition}'. Must be one of 'longest', 'tightest', or 'loosest'."
            )
        self.probe()
        plateaus = self.find_natural_plateaus()
        if not plateaus:
            warnings.warn(
                "No natural plateaus found in the tolerance probe. Consider a larger sample_size, more probe_steps or a larger plateau_threshold.\nReturning the geometric mean of the all-different and all-same tolerances as fallback.",
                stacklevel=2,
            )
            return float(
                np.sqrt(self.all_different_tolerance * self.all_same_tolerance)
            )
        if condition == "tightest":
            chosen = plateaus[0]
        elif condition == "loosest":
            chosen = plateaus[-1]
        else:
            chosen = max(plateaus, key=lambda p: p.steps)
        logger.info("Chose plateau %s (condition=%r).", chosen, condition)
        return chosen.tolerance
