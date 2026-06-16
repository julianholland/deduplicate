Tolerance Calculators
=====================

Tolerance calculators automatically find a ``tolerance`` value for a given
dataset and DDA.  Run the calculator once, then assign the result to
``dda.tolerance`` before calling ``deduplicate()`` or ``duplicate_check()``.


PerturbedDatasetReclustering
-----------------------------

Perturbs each original vector ``perturbations_per_vector`` times with Gaussian
noise of scale ``perturbation_scale``, then binary-searches for the tolerance
that recovers the original unique-vector count in the perturbed dataset.
``"average"`` mode averages the loose and tight bounds; ``"loose"`` returns the
highest valid tolerance; ``"tight"`` returns the lowest.

.. autoclass:: deduplicate_lib.plugins.tolerance_calculators.perturbed_dataset_reclustering.PerturbedDatasetReclustering
   :members: calculate_tolerance
   :show-inheritance:


NaturalTolerancePlateauProbe
-----------------------------

No perturbed data needed.  Sweeps from the all-same tolerance bound to the
all-different bound, records the unique-structure count at each step, and picks
the midpoint of the longest plateau — a region where the clustering is stable
and insensitive to small changes in tolerance.

.. autoclass:: deduplicate_lib.plugins.tolerance_calculators.natural_tolerance_plateau_probe.NaturalTolerancePlateauProbe
   :members: calculate_tolerance
   :show-inheritance:
