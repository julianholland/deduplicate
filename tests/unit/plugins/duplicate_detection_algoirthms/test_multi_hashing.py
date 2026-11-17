import copy
import os
import re
import subprocess
import sys

import numpy as np
import pytest

from deduplicate_lib.plugins.duplicate_detection_algorithms.multi_hashing import (
    MultiHashing,
    fast_quantise_and_hash,
)

from .reference_multi_hashing import (
    reference_clash_vector,
    reference_unique_mask,
)


def clustered_dataset(seed, n_clusters=15, per_cluster=6, dim=8, spread=0.02):
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-5, 5, size=(n_clusters, dim))
    data = np.repeat(centres, per_cluster, axis=0)
    data += rng.normal(scale=spread, size=data.shape)
    return data[rng.permutation(len(data))]


def test_fast_quantise_and_hash_shape_and_dtype():
    vectors = np.array([[1.01, 2.01], [5.0, 6.0], [1.0, 2.0]])
    hashes = fast_quantise_and_hash(vectors, np.array([0.1, 0.2, 0.3]), 0.1)
    assert hashes.shape == (3, 3)
    assert hashes.dtype == np.uint64


def test_fast_quantise_and_hash_equal_iff_same_grid_point():
    tolerance = 0.1
    perturbation_array = np.array([1.0])
    vectors = np.array(
        [
            [1.01, 2.01],  # rounds to (1.0, 2.0)
            [0.99, 1.98],  # rounds to (1.0, 2.0)
            [1.2, 2.0],  # rounds to (1.2, 2.0)
            [-0.001, 0.0],  # rounds to (-0.0, 0.0): must equal (0.0, 0.0)
            [0.0, 0.0],
        ]
    )
    hashes = fast_quantise_and_hash(vectors, perturbation_array, tolerance)[0]
    assert hashes[0] == hashes[1]
    assert hashes[0] != hashes[2]
    assert hashes[3] == hashes[4]


def test_fast_quantise_and_hash_is_deterministic_across_processes():
    """Unlike Python's hash(), the hashes must not depend on PYTHONHASHSEED."""
    script = (
        "import numpy as np\n"
        "from deduplicate_lib.plugins.duplicate_detection_algorithms.multi_hashing "
        "import fast_quantise_and_hash\n"
        "v = np.arange(12, dtype=float).reshape(3, 4) / 7\n"
        "print(fast_quantise_and_hash(v, np.array([0.5, 1.5]), 0.1).tolist())\n"
    )
    outputs = set()
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(result.stdout)
    assert len(outputs) == 1


def test_str_representation():
    dm = MultiHashing(
        tolerance=0.1,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0]]),
        distance_matrix=np.array([]),
    )
    assert (
        str(dm)
        == "MultiHashing(tolerance=0.1, perturbations=200, sigma_acceptance_threshold=1)"
    )


def test_set_perturbation_array(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.set_perturbation_array()
    assert dda.perturbation_array.shape[0] == dda.perturbations


def test_round_to_tolerance(multi_hashing_dda):
    dda = multi_hashing_dda
    rounded_vector = dda.round_to_tolerance()
    assert np.allclose(rounded_vector, np.array([1.0, 2.0]))


def test_create_hash_vector(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.set_perturbation_array()
    hash_vector = dda.create_hash_vector()
    assert hash_vector.shape == (dda.perturbations,)
    assert hash_vector.dtype == np.uint64
    assert np.array_equal(hash_vector, dda.create_hash_vector(dda.input_vector))


def test_compute_hash_tables(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.set_perturbation_array()
    dda.compute_hash_tables()
    assert len(dda.sorted_hashes) == dda.perturbations
    for table in dda.sorted_hashes:
        assert table.dtype == np.uint64
        assert np.all(np.diff(table.astype(np.float64)) > 0)  # sorted, distinct
        assert 0 < len(table) <= dda.vector_count


def test_compute_hash_tables_without_setting_perturbation_array(multi_hashing_dda):
    dda = multi_hashing_dda
    with pytest.warns(
        UserWarning,
        match="Perturbation array shape does not match expected number of perturbations; recomputing. \nAssign the perturbation array prior duplication checks to avoid this warning.",
    ):
        dda.compute_hash_tables()
    assert len(dda.sorted_hashes) == dda.perturbations


def test_compute_hash_tables_consistency(multi_hashing_dda):
    dda1 = multi_hashing_dda
    dda1.set_perturbation_array()
    dda1.compute_hash_tables()
    dda2 = copy.deepcopy(dda1)
    dda2.compute_hash_tables()
    for table_1, table_2 in zip(dda1.sorted_hashes, dda2.sorted_hashes):
        assert np.array_equal(table_1, table_2)


def test_compute_hash_tables_different_seeds(multi_hashing_dda):
    dda1 = copy.deepcopy(multi_hashing_dda)
    dda1.seed = 803
    dda1.set_perturbation_array()
    dda1.compute_hash_tables()

    dda2 = copy.deepcopy(multi_hashing_dda)
    dda2.seed = 8  # different seed
    dda2.set_perturbation_array()
    dda2.compute_hash_tables()

    assert not np.array_equal(dda1.sorted_hashes[0], dda2.sorted_hashes[0])


def test_compute_hash_tables_different_perturbations(multi_hashing_dda):
    dda1 = copy.deepcopy(multi_hashing_dda)
    dda1.perturbations = 200
    dda1.set_perturbation_array()
    dda1.compute_hash_tables()

    dda2 = copy.deepcopy(multi_hashing_dda)
    dda2.perturbations = 300  # different number of perturbations
    dda2.set_perturbation_array()
    dda2.compute_hash_tables()

    assert len(dda1.sorted_hashes) == 200
    assert len(dda2.sorted_hashes) == 300


def test_compute_hash_tables_ignores_input_vector(multi_hashing_dda):
    # only the dataset vectors go into the hash tables, not the input vector
    dda1 = copy.deepcopy(multi_hashing_dda)
    dda1.set_perturbation_array()
    dda1.compute_hash_tables()

    dda2 = copy.deepcopy(multi_hashing_dda)
    dda2.input_vector = np.array([10.0, 20.0])
    dda2.set_perturbation_array()
    dda2.compute_hash_tables()

    for table_1, table_2 in zip(dda1.sorted_hashes, dda2.sorted_hashes):
        assert np.array_equal(table_1, table_2)


def test_compute_hash_tables_different_dataset_arrays(multi_hashing_dda):
    dda1 = copy.deepcopy(multi_hashing_dda)
    dda1.set_perturbation_array()
    dda1.compute_hash_tables()

    dda2 = copy.deepcopy(multi_hashing_dda)
    dda2.set_dataset_array(np.array([[10.0, 20.0], [11.0, 21.0], [9.0, 19.0]]))
    dda2.set_perturbation_array()
    dda2.compute_hash_tables()

    assert not np.array_equal(dda1.sorted_hashes[0], dda2.sorted_hashes[0])


def test_compute_hash_tables_in_batches_and_chunks(monkeypatch):
    """Building the tables in small perturbation batches and row chunks gives the same result."""
    from deduplicate_lib.core import resources

    data = clustered_dataset(seed=1)
    dda = MultiHashing(tolerance=0.1, perturbations=50, dataset_array=data)
    dda.pre_dda_processing()
    expected_counts = dda.unique_counts.copy()
    expected_tables = [t.copy() for t in dda.sorted_hashes]

    monkeypatch.setattr(resources, "perturbation_batch", lambda *_: 3)
    monkeypatch.setattr(resources, "chunk_rows", lambda *_: 7)
    dda.compute_hash_tables()
    assert np.array_equal(dda.unique_counts, expected_counts)
    for table, expected in zip(dda.sorted_hashes, expected_tables):
        assert np.array_equal(table, expected)


def test_initialize_hash_tables(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.set_perturbation_array()
    dda.initialize_hash_tables()
    assert len(dda.sorted_hashes) == dda.perturbations
    assert all(len(table) == 0 for table in dda.sorted_hashes)
    assert not np.any(dda.unique_counts)


def test_duplicate_check(multi_hashing_dda):
    dda = multi_hashing_dda

    dda.set_perturbation_array()  # Ensure perturbation array is set before duplication check
    dda.compute_hash_tables()  # Ensure hash tables are created before duplication check
    assert dda.duplicate_check()

    dda.input_vector = np.array([16.0, 200.0])
    assert not dda.duplicate_check()


def test_duplicate_check_rebuilds_tables_if_perturbations_changed(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.pre_dda_processing()
    dda.perturbations = 7
    dda.set_perturbation_array()
    assert dda.duplicate_check()
    assert len(dda.sorted_hashes) == 7


def test_set_acceptance_threshold(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.sigma_accepatnce_threshold = 2
    dda.perturbations = 200
    dda.set_acceptance_threshold()
    assert dda.acceptance_threshold == 0.954499736103642
    dda.perturbations = 5


def test_set_acceptance_threshold_bad_sigma(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.sigma_accepatnce_threshold = 5  # not in the sigma_dict
    with pytest.raises(
        ValueError,
        match=r"Sigma acceptance threshold must be an integer between 1 and 4 inclusive.",
    ):
        dda.set_acceptance_threshold()


def test_set_acceptance_threshold_warning(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.sigma_accepatnce_threshold = 2
    dda.perturbations = 5
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Sigma acceptance threshold of 2 corresponds to an acceptance threshold of 0.954499736103642 which may be too high for the number of perturbations (5) and could lead to false positives. Consider lowering the sigma acceptance threshold or increasing the number of perturbations to over 22."
        ),
    ):
        dda.set_acceptance_threshold()


def test_get_dataset_unique_structures(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.perturbations = 5
    dda.set_dataset_array(np.array([[1.0, 2.0], [1.001, 2.001], [10.0, 20.0]]))

    dda.set_perturbation_array()
    dda.compute_hash_tables()
    unique_count = dda.get_dataset_unique_structures()
    assert unique_count == 2
    assert isinstance(unique_count, int)


def test_pre_dda_processing(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.set_dataset_array(np.array([]))
    dda.pre_dda_processing()
    assert dda.perturbation_array.shape[0] == dda.perturbations
    assert all(len(table) == 0 for table in dda.sorted_hashes)
    assert dda.get_dataset_unique_structures() == 0


def test_add_input_vector_to_dda(multi_hashing_dda):
    dda = multi_hashing_dda
    initial_dataset_size = dda.get_filled_dataset_array().shape[0]
    dda.pre_dda_processing()
    dda.add_input_vector_to_dda()
    assert dda.get_filled_dataset_array().shape[0] == initial_dataset_size + 1
    assert np.array_equal(dda.get_filled_dataset_array()[-1], dda.input_vector)
    # the input vector equals dataset row 0, so it is never first in its bucket
    assert dda.unique_counts[dda.vector_count - 1] == 0
    assert dda.duplicate_check()


def test_add_input_vector_to_dda_without_prior_pre_dda_processing():
    dda = MultiHashing(
        tolerance=0.01,
        perturbations=5,
        input_vector=np.array([10.0, 20.0]),
        dataset_array=np.array([[1.0, 2.0], [1.01, 2.01], [0.0, 2.0]]),
    )
    initial_dataset_size = dda.get_filled_dataset_array().shape[0]
    dda.add_input_vector_to_dda()
    assert dda.get_filled_dataset_array().shape[0] == initial_dataset_size + 1
    assert np.array_equal(dda.get_filled_dataset_array()[-1], dda.input_vector)
    # a new, distant vector is first in its bucket for every perturbation
    assert dda.unique_counts[dda.vector_count - 1] == dda.perturbations
    assert dda.duplicate_check()  # now it is in the dataset


def test_add_input_vector_hashes_without_input_vector(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.set_dataset_array(np.array([[1.0, 2.0], [1.001, 2.001], [10.0, 20.0]]))
    dda.set_perturbation_array()
    dda.compute_hash_tables()
    dda.add_input_vector_hashes(input_vector=None)
    assert dda.unique_counts[dda.vector_count] == 0  # input vector equals row 0


def test_get_uniqueness_score(multi_hashing_dda):
    dda = multi_hashing_dda
    dda.set_perturbation_array()
    dda.compute_hash_tables()
    dda.input_vector = dda.dataset_array[
        0
    ]  # identical to first dataset vector, should be classified as duplicate
    uniqueness_score = dda.get_uniqueness_score()

    assert uniqueness_score == 0.0

    dda.input_vector = np.array(
        [10.0, 20.0]
    )  # different from all dataset vectors, should be classified as unique
    uniqueness_score = dda.get_uniqueness_score()
    assert uniqueness_score > 0.0 and uniqueness_score <= 1.0


def test_duplicate_check_rebuilds_hash_tables_on_tolerance_change():
    dda = MultiHashing(
        tolerance=0.01,
        perturbations=5,
        input_vector=np.array([10.0, 20.0]),
        dataset_array=np.array([[1.0, 2.0], [1.01, 2.01], [0.0, 2.0]]),
    )

    # first build, with the initial tolerance
    first_result = dda.duplicate_check()
    assert dda._dirty is False

    # change tolerance to something which should change clash behaviour
    dda.tolerance = 50.0
    second_result = dda.duplicate_check()

    # force a rebuild from scratch with the same (new) tolerance
    dda._dirty = True
    third_result = dda.duplicate_check()

    assert second_result == third_result
    # sanity check that the tolerance change actually mattered vs the first result
    assert first_result != second_result


def test_duplicate_check_regenerates_perturbation_array_on_seed_change():
    dda = MultiHashing(
        tolerance=0.01,
        perturbations=5,
        seed=803,
        input_vector=np.array([1.0, 2.0]),
        dataset_array=np.array([[1.0, 2.0], [1.01, 2.01], [10.0, 20.0]]),
    )

    dda.duplicate_check()
    perturbation_array_before = dda.perturbation_array.copy()

    dda.seed = 8  # different seed
    dda.duplicate_check()
    perturbation_array_after = dda.perturbation_array

    assert not np.array_equal(perturbation_array_before, perturbation_array_after)


# --- equivalence with the dict-based implementation of deduplicate_lib <= 0.1.3 ---


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("tolerance", [0.01, 0.05, 0.2, 1.0, 5.0])
@pytest.mark.parametrize("sigma", [1, 2, 3])
def test_unique_mask_matches_reference(seed, tolerance, sigma):
    data = clustered_dataset(seed)
    dda = MultiHashing(
        tolerance=tolerance,
        perturbations=60,
        seed=seed,
        sigma_accepatnce_threshold=sigma,
        dataset_array=data,
    )
    dda.get_dataset_unique_structures()
    expected = reference_unique_mask(
        data, dda.perturbation_array, tolerance, dda.acceptance_threshold
    )
    assert np.array_equal(dda.get_unique_vector_mask(), expected)


@pytest.mark.parametrize("tolerance", [0.05, 0.5])
def test_clash_vector_matches_reference(tolerance):
    data = clustered_dataset(seed=3)
    dda = MultiHashing(tolerance=tolerance, perturbations=40, dataset_array=data)
    dda.pre_dda_processing()
    rng = np.random.default_rng(3)
    probes = np.vstack([data[:5] + rng.normal(scale=0.02, size=(5, 8)), data[5:8]])
    for probe in probes:
        dda.input_vector = probe
        expected = reference_clash_vector(
            data, probe, dda.perturbation_array, tolerance
        )
        assert np.array_equal(dda._clash_vector(dda.create_hash_vector()), expected)
        assert dda.get_uniqueness_score() == 1 - expected.mean()
        assert dda.duplicate_check() == (expected.mean() >= dda.acceptance_threshold)


def test_streamed_appends_match_batch_build():
    """Appending vectors one at a time (through several pending-hash merges)
    gives the same unique mask and tables as building from the full dataset."""
    data = clustered_dataset(seed=4, n_clusters=1500, per_cluster=2, dim=4)
    streamed = MultiHashing(tolerance=0.1, perturbations=20, input_vector=data[0])
    for vector in data:
        streamed.input_vector = vector
        streamed.add_input_vector_to_dda()
    assert any(len(table) > 0 for table in streamed.sorted_hashes)  # merged

    batch = MultiHashing(tolerance=0.1, perturbations=20, dataset_array=data)
    streamed.get_dataset_unique_structures()
    batch.get_dataset_unique_structures()
    assert np.array_equal(
        streamed.get_unique_vector_mask(), batch.get_unique_vector_mask()
    )
    expected = reference_unique_mask(
        data, batch.perturbation_array, 0.1, batch.acceptance_threshold
    )
    assert np.array_equal(batch.get_unique_vector_mask(), expected)


# --- user-supplied perturbation arrays ---

PERT_DATA = clustered_dataset(seed=21, n_clusters=12, per_cluster=4, dim=5)


def test_custom_perturbation_array_survives_build():
    custom = np.full(10, 0.5)
    dda = MultiHashing(
        tolerance=0.1,
        perturbations=10,
        pertrubation_array=custom,
        dataset_array=PERT_DATA,
    )
    dda.get_dataset_unique_structures()
    assert np.array_equal(dda.perturbation_array, custom)


def test_seed_change_regenerates_generated_array():
    dda = MultiHashing(tolerance=0.1, perturbations=10, seed=1, dataset_array=PERT_DATA)
    dda.get_dataset_unique_structures()
    first = dda.perturbation_array.copy()
    dda.seed = 2
    dda.get_dataset_unique_structures()
    assert not np.array_equal(dda.perturbation_array, first)
    expected = np.abs(np.random.default_rng(2).normal(size=10))
    assert np.array_equal(dda.perturbation_array, expected)
    assert np.array_equal(dda._hashed_perturbation_array, expected)


def test_seed_change_does_not_override_custom_array():
    custom = np.full(10, 0.5)
    dda = MultiHashing(
        tolerance=0.1,
        perturbations=10,
        pertrubation_array=custom,
        dataset_array=PERT_DATA,
    )
    dda.get_dataset_unique_structures()
    dda.seed = 99
    dda.get_dataset_unique_structures()
    assert np.array_equal(dda.perturbation_array, custom)


def test_new_perturbation_array_after_build_triggers_rebuild():
    dda = MultiHashing(tolerance=0.1, perturbations=10, seed=3, dataset_array=PERT_DATA)
    dda.get_dataset_unique_structures()
    # a degenerate array (all zeros) puts every vector in one bucket
    dda.perturbation_array = np.zeros(10)
    assert dda.get_dataset_unique_structures() == 1
    # and a fresh DDA with that array agrees
    other = MultiHashing(
        tolerance=0.1,
        perturbations=10,
        pertrubation_array=np.zeros(10),
        dataset_array=PERT_DATA,
    )
    assert other.get_dataset_unique_structures() == 1
