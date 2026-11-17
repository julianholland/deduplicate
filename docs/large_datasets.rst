Large datasets
==============

Datasets too large to hold in memory (millions of vectors) can be deduplicated
with :class:`~deduplicate_lib.plugins.duplicate_detection_algorithms.multi_hashing.MultiHashing`
straight from disk.  Pass ``dataset`` instead of ``dataset_array``; the vectors
are then read in chunks and never loaded whole.

.. code-block:: python

    from deduplicate_lib.plugins.duplicate_detection_algorithms.multi_hashing import MultiHashing

    dda = MultiHashing(
        dataset="descriptors.npy",   # or "descriptors.h5" (needs deduplicate_lib[hdf5])
        tolerance=0.05,
        perturbations=200,
        memory_limit="8GB",          # default "auto": half of the available memory
        n_threads=8,                 # default: numba's setting
        progress=True,               # progress bars (needs deduplicate_lib[progress])
    )

    n_unique = dda.get_dataset_unique_structures()
    mask = dda.get_unique_vector_mask()      # boolean, one entry per vector
    indices = dda.get_unique_vector_indices() # integer indices of the unique vectors
    unique = dda.deduplicate()               # writes descriptors_unique.npy


Where the data can live
-----------------------

``dataset`` takes a path or a :class:`~deduplicate_lib.core.vector_store.VectorStore`:

==========================  =================================================
``"x.npy"``                 memory-mapped, read-only
``"x.h5"`` / ``"x.hdf5"``   the ``"vectors"`` dataset (``HDF5Store(path, key=...)``
                            for another name); appendable if resizable
``CallableStore``           anything else, e.g. a structure database
==========================  =================================================

``CallableStore`` adapts any source that can return a range of rows:

.. code-block:: python

    from deduplicate_lib.core.vector_store import CallableStore

    store = CallableStore(
        n_vectors=db.count(),
        dim=128,
        fetch=lambda start, stop: db.descriptors(start, stop),  # rows [start, stop)
    )
    dda = MultiHashing(dataset=store, tolerance=0.05)

Rows are always fetched in order and in chunks, so ``fetch`` can read a batch per
call.


What you get back
-----------------

* ``get_unique_vector_mask()`` — boolean mask over the dataset (7 MB for 7M vectors).
* ``get_unique_vector_indices()`` — the integer indices of the unique vectors.
* ``deduplicate(out=None)`` — the unique vectors.  For an in-memory dataset with
  no ``out``, an array.  Otherwise they are streamed to a file and the file is
  returned, opened as a store (``.path`` says where).  By default the file
  mirrors the input: ``x.npy`` → ``x_unique.npy``, ``x.h5`` → ``x_unique.h5``.
  The format follows the extension of ``out``, so ``deduplicate(out="u.h5")``
  converts a ``.npy`` input.  A ``CallableStore`` has no file to name the output
  after, so it needs ``out``.


Memory
------

``memory_limit`` bounds the working memory:

* a quarter for the chunk of vectors being hashed,
* half for the hashes being built (one ``uint64`` per vector per perturbation;
  perturbations are processed in batches that fit, reading the dataset once per
  batch),
* the per-perturbation hash tables are kept only if they fit in a quarter.

At 7M vectors × 200 perturbations the hash tables take up to 11 GB.  Below a
``memory_limit`` of 45 GB they are not kept in memory:
``get_dataset_unique_structures()`` and ``deduplicate()`` work as usual, but
``duplicate_check()``, ``get_uniqueness_score()`` and appending vectors raise
``MemoryError`` unless the tables are kept on disk with ``index_dir`` (below).

:class:`~deduplicate_lib.plugins.duplicate_detection_algorithms.distance_matrix.DistanceMatrix`
needs an n × n matrix in memory (4.9 × 10¹³ entries for 7M vectors), so it
refuses on-disk datasets and matrices larger than ``memory_limit``.


Keeping the hash tables: ``index_dir``
--------------------------------------

Pass ``index_dir`` to keep the hash tables on disk.  They are memory-mapped
instead of held in memory, so ``duplicate_check()`` and appends work at any
size, and the work is never repeated:

.. code-block:: python

    dda = MultiHashing(dataset="descriptors.h5", tolerance=0.05,
                       index_dir="descriptors.dedup", memory_limit="8GB")
    dda.get_dataset_unique_structures()   # builds the index (minutes for millions)

    # later, in another process: opens the index instead of hashing
    dda = MultiHashing(dataset="descriptors.h5", tolerance=0.05,
                       index_dir="descriptors.dedup")
    dda.input_vector = candidate
    dda.duplicate_check()                 # milliseconds

    dda.add_vectors(new_vectors)          # appends to the HDF5 file and the tables
    dda.flush()                           # saves the tables

What the index does for you:

* **Reuse.** An index is reused when the dataset and the hashing parameters
  (``tolerance``, ``perturbations``, ``seed``/``pertrubation_array``) match.
  ``sigma_accepatnce_threshold`` is applied at query time, so it can change
  freely.  Anything else raises ``IndexMismatchError`` naming what differs.
  Each tolerance needs its own ``index_dir``.  To check that the dataset is
  the one the index was built from, opening an index reads the indexed rows
  once and compares their checksums (far cheaper than hashing them again).
* **Resume.** The build commits after every perturbation batch.  If it is killed
  (e.g. by a job time limit), running the same code again hashes only the
  perturbations still missing.
* **Appends.** ``add_vectors()`` and ``add_input_vector_to_dda()`` add to the
  tables; ``flush()`` saves them (it also happens automatically when the pending
  hashes grow large).  Vectors appended to an HDF5 dataset but never flushed,
  or added to the file by other code, are hashed and saved the next time the
  index is opened.
* **Crash safety.** ``metadata.json`` is the commit point and is replaced
  atomically; table files are never overwritten, only superseded, so an
  interrupted write leaves the previous commit intact.  Files are fsynced before
  each commit.  On filesystems where that is slow, setting
  ``deduplicate_lib.core.hash_index.FSYNC = False`` keeps the guarantee against
  killed processes but not against power loss.

Disk use is one ``uint64`` per distinct hash per perturbation: 3.7 GB for
7M vectors (1.75M distinct groups) × 200 perturbations.  One process should
write to an index at a time.


Choosing a tolerance
--------------------

The tolerance calculators tune on a sample of the dataset, read through
``dda.store``, so they work the same for in-memory and out-of-core datasets.
Every search runs on a separate in-memory copy of the DDA
(``dda.tuning_copy(sample)``): the DDA you pass in, its store and its
``index_dir`` are never touched.  Tune, then run once on the full dataset at
that tolerance:

.. code-block:: python

    from deduplicate_lib.plugins.tolerance_calculators.natural_tolerance_plateau_probe import (
        NaturalTolerancePlateauProbe,
    )

    dda = MultiHashing(dataset="descriptors.h5", perturbations=200)
    tc = NaturalTolerancePlateauProbe(dda, sample_size=5000)
    dda.tolerance = tc.calculate_tolerance()  # or "tightest" / "loosest"
    dda.get_dataset_unique_structures()

    for plateau in tc.find_natural_plateaus():  # every natural clustering scale
        print(plateau.start_tolerance, plateau.end_tolerance, plateau.unique_vectors)

``sample_size`` sets how many vectors are tuned on, whatever the dataset size.
``sampling`` sets how they are drawn:

==========  ==================================================================
``blocks``  contiguous blocks of ``block_rows`` (100) rows from random places;
            default for ``NaturalTolerancePlateauProbe``
``random``  uniformly random rows; default for ``PerturbedDatasetReclustering``
``head``    the first ``sample_size`` rows
==========  ==================================================================

Why blocks: a uniform sample of *k* of *n* vectors keeps both members of a
near-duplicate pair with probability about (*k*/*n*)², so on a large dataset it
contains almost no duplicates and the plateau probe sees none.  Datasets often
store related vectors next to each other (consecutive frames of a trajectory,
structures from one run), and blocks keep those together.  If your duplicates
are scattered through the file, sample a larger ``sample_size``.
``PerturbedDatasetReclustering`` makes its own duplicates (noisy copies of each
sampled vector) and assumes the sampled vectors are distinct, so it samples
uniformly.
