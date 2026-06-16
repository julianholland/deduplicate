Core API
========

The typical DDA lifecycle is: construct the DDA (optionally passing
``dataset_array`` or calling ``set_dataset_array()`` afterwards) → set
``input_vector`` → call ``duplicate_check()`` or
``add_input_vector_to_dda()`` for streaming ingestion → call
``deduplicate()`` to retrieve the full set of unique vectors.

.. autoclass:: deduplicate_lib.core.duplicate_detection_algorithm.DuplicateDetectionAlgorithm
   :members: deduplicate, duplicate_check, add_input_vector_to_dda, set_dataset_array, get_dataset_unique_structures
   :show-inheritance:
