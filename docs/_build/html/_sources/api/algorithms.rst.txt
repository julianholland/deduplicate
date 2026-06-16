Duplicate Detection Algorithms
==============================

Two algorithms are provided.  Choose based on dataset size and whether exact
results are required.


DistanceMatrix
--------------

Exact algorithm.  Builds the full N×N pairwise distance matrix and marks every
vector whose nearest neighbour is within ``tolerance`` as a duplicate.  O(N²)
time and space; a good choice for datasets up to ~5 000 vectors.  Supports all
four distance metrics: ``"euclidean"``, ``"manhattan"``, ``"cosine"``, and
``"hamming"``.

.. autoclass:: deduplicate_lib.plugins.duplicate_detection_algorithms.distance_matrix.DistanceMatrix
   :members:
   :show-inheritance:


MultiHashing
------------

Approximate algorithm.  O(N · perturbations) time; scales well to large
datasets.  Restricted to ``"hamming"`` distance.  Accuracy is controlled by
``perturbations`` (more is better) and ``sigma_accepatnce_threshold`` (higher
reduces false positives but requires more perturbations to be reliable).

.. autoclass:: deduplicate_lib.plugins.duplicate_detection_algorithms.multi_hashing.MultiHashing
   :members:
   :show-inheritance:

.. note::

   ``sigma_accepatnce_threshold`` selects the acceptance band used to declare
   a hash-collision a duplicate:

   ==============================  ================
   ``sigma_accepatnce_threshold``  Acceptance band
   ==============================  ================
   1 (default)                     68.3%
   2                               95.4%
   3                               99.7%
   4                               99.99%
   ==============================  ================

   Higher values reduce false positives but require more perturbations to be
   statistically reliable.
