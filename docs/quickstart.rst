Quick Start
===========

Batch deduplication
-------------------

Use :class:`~deduplicate_lib.plugins.duplicate_detection_algorithms.distance_matrix.DistanceMatrix`
to load a dataset and return only the unique vectors in a single call.

.. code-block:: python

    from deduplicate_lib.plugins.duplicate_detection_algorithms.distance_matrix import DistanceMatrix
    import numpy as np

    dataset = np.array([[1.0, 2.0], [1.01, 2.01], [5.0, 6.0]])
    dda = DistanceMatrix(tolerance=0.1, dataset_array=dataset)
    unique = dda.deduplicate()
    print(unique)


Single-vector duplicate check
------------------------------

``duplicate_check()`` is fast — it computes distances only for ``input_vector``
rather than rebuilding the full matrix.

.. code-block:: python

    dda.input_vector = np.array([1.0, 2.0])
    is_duplicate = dda.duplicate_check()
    print(is_duplicate)  # True — matches first row within tolerance 0.1


Streaming / incremental ingestion
----------------------------------

``add_input_vector_to_dda()`` appends ``input_vector`` to the dataset and
incrementally updates auxiliary structures, avoiding a full rebuild on every
insertion.

.. code-block:: python

    from deduplicate_lib.plugins.duplicate_detection_algorithms.multi_hashing import MultiHashing

    dda = MultiHashing(tolerance=0.1, perturbations=200)
    for vec in incoming_vectors:
        dda.input_vector = vec
        if not dda.duplicate_check():
            dda.add_input_vector_to_dda()


Automatic tolerance tuning
---------------------------

When you do not know a good ``tolerance`` value, use a
:class:`~deduplicate_lib.plugins.tolerance_calculators.perturbed_dataset_reclustering.PerturbedDatasetReclustering`
to find one automatically, then assign it back to the DDA.

.. code-block:: python

    from deduplicate_lib.plugins.tolerance_calculators.perturbed_dataset_reclustering import PerturbedDatasetReclustering

    dda = DistanceMatrix(dataset_array=dataset)
    tc = PerturbedDatasetReclustering(
        duplicate_detection_algorithm_object=dda,
        perturbations_per_vector=3,
        perturbation_scale=0.05,
    )
    tc.create_perturbed_dataset()
    dda.tolerance = tc.calculate_tolerance()
    unique = dda.deduplicate()
