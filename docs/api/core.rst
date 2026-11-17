Core API
========

The typical DDA lifecycle is: construct the DDA (optionally passing
``dataset_array`` or calling ``set_dataset_array()`` afterwards) → set
``input_vector`` → call ``duplicate_check()`` or
``add_input_vector_to_dda()`` for streaming ingestion → call
``deduplicate()`` to retrieve the full set of unique vectors.

.. autoclass:: deduplicate_lib.core.duplicate_detection_algorithm.DuplicateDetectionAlgorithm
   :members: deduplicate, duplicate_check, add_input_vector_to_dda, set_dataset_array, set_dataset_store, store, is_out_of_core, get_dataset_unique_structures, get_unique_vector_mask, get_unique_vector_indices
   :show-inheritance:


Vector stores
-------------

Where a dataset's vectors live.  Pass a path or a store as ``dataset`` to use an
out-of-core dataset; see :doc:`../large_datasets`.

.. automodule:: deduplicate_lib.core.vector_store
   :members: VectorStore, InMemoryStore, NpyStore, HDF5Store, CallableStore, open_store
   :show-inheritance:


Hash index
----------

.. automodule:: deduplicate_lib.core.hash_index
   :members: HashIndex, IndexMismatchError


Resources
---------

.. automodule:: deduplicate_lib.core.resources
   :members: parse_memory_limit, chunk_rows, perturbation_batch
