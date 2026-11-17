import numpy as np
import pytest

from deduplicate_lib.plugins.tolerance_calculators.perturbed_dataset_reclustering import (
    PerturbedDatasetReclustering,
)

N_VECTORS = 40


@pytest.fixture
def spread_data():
    return np.random.default_rng(11).uniform(0.0, 1.0, size=(N_VECTORS, 8))


def test_str_representation(distance_matrix_dda):
    dda = distance_matrix_dda
    assert (
        str(
            PerturbedDatasetReclustering(
                dda, perturbations_per_vector=1000, perturbation_scale=0.1
            )
        )
        == f"PerturbedDatasetReclustering(perturbations_per_vector=1000, perturbation_scale=0.1, dda={str(dda).split('(')[0]})"
    )


def test_defaults(distance_matrix_dda):
    tc = PerturbedDatasetReclustering(distance_matrix_dda)
    assert tc.perturbations_per_vector == 5
    assert tc.perturbation_scale == 0.1
    assert tc.target_unique_vectors is None
    assert tc.target_unique_vectors_threshold == "average"
    assert tc.sample_size == 500
    assert tc.sampling == "random"


def test_perturbation_args_are_keyword_only(distance_matrix_dda):
    with pytest.raises(TypeError):
        PerturbedDatasetReclustering(distance_matrix_dda, 5)  # type: ignore


@pytest.mark.parametrize("ppv", [0, -1, 2.5])
def test_invalid_perturbations_per_vector(distance_matrix_dda, ppv):
    with pytest.raises(ValueError, match="perturbations_per_vector"):
        PerturbedDatasetReclustering(distance_matrix_dda, perturbations_per_vector=ppv)


def test_invalid_threshold_raises_in_init(distance_matrix_dda):
    with pytest.raises(
        ValueError,
        match=r"Invalid target_unique_vectors_threshold: invalid_mode\. Must be 'average', 'loose', or 'tight'\.",
    ):
        PerturbedDatasetReclustering(
            distance_matrix_dda, target_unique_vectors_threshold="invalid_mode"
        )


def test_initialize_with_target_unique_vectors(distance_matrix_dda):
    tc = PerturbedDatasetReclustering(distance_matrix_dda, target_unique_vectors=2)
    assert tc.target_unique_vectors == 2


def test_tuning_array_layout(distance_matrix_dda, spread_data):
    distance_matrix_dda.set_dataset_array(spread_data)
    tc = PerturbedDatasetReclustering(
        distance_matrix_dda,
        perturbations_per_vector=4,
        perturbation_scale=0.01,
        sample_size=10,
    )
    tc._prepare_tuning()
    tuning = tc.tuning_array
    assert tuning.shape == (40, 8)
    originals = tuning[::4]
    # originals are exact rows of the dataset; copies are noisy versions of them
    assert all(any(np.array_equal(o, row) for row in spread_data) for o in originals)
    for i in range(10):
        copies = tuning[4 * i + 1 : 4 * i + 4] - originals[i]
        assert (np.abs(copies) > 0).all()
        assert np.abs(copies).max() < 0.1


@pytest.mark.parametrize("dda_fixture", ["distance_matrix_dda", "multi_hashing_dda"])
def test_calculate_tolerance(request, dda_fixture, spread_data):
    dda = request.getfixturevalue(dda_fixture)
    dda.set_dataset_array(spread_data)
    original_tolerance = dda.tolerance
    original_array = dda.get_filled_dataset_array().copy()

    def run(threshold):
        tc = PerturbedDatasetReclustering(
            dda,
            perturbations_per_vector=5,
            perturbation_scale=0.01,
            binary_search_steps=20,
            sample_size=20,
            target_unique_vectors_threshold=threshold,
        )
        tolerance = tc.calculate_tolerance()
        assert isinstance(tolerance, float)
        assert tolerance > 0.0
        if threshold != "average":  # the mean of the two need not be exact (hashing)
            # the copies collapse onto their originals at the returned tolerance
            tuning = dda.tuning_copy(tc.tuning_array)
            tuning.tolerance = tolerance
            assert tuning.get_dataset_unique_structures() == 20
        return tolerance

    average, loose, tight = run("average"), run("loose"), run("tight")
    assert tight < average < loose

    # the user's DDA was never touched
    assert dda.tolerance == original_tolerance
    assert np.array_equal(dda.get_filled_dataset_array(), original_array)


def test_calculate_tolerance_is_deterministic(distance_matrix_dda, spread_data):
    distance_matrix_dda.set_dataset_array(spread_data)

    def run(seed):
        return PerturbedDatasetReclustering(
            distance_matrix_dda,
            perturbation_scale=0.01,
            sample_size=15,
            seed=seed,
        ).calculate_tolerance()

    assert run(1) == run(1)


@pytest.mark.parametrize("dda_fixture", ["distance_matrix_dda", "multi_hashing_dda"])
def test_calculate_tolerance_with_insufficient_binary_search_steps(
    request, dda_fixture, spread_data
):
    dda = request.getfixturevalue(dda_fixture)
    dda.set_dataset_array(spread_data)
    tc = PerturbedDatasetReclustering(
        dda,
        perturbations_per_vector=5,
        perturbation_scale=0.01,
        sample_size=20,
        binary_search_steps=1,
    )
    with pytest.warns(UserWarning, match="No exact tolerance found for target of 20"):
        tolerance = tc.calculate_tolerance()
    # one step only tries the upper bound
    assert tolerance == tc.tolerance_upper_bound()


def test_calculate_tolerance_with_explicit_target(distance_matrix_dda, spread_data):
    distance_matrix_dda.set_dataset_array(spread_data)
    tc = PerturbedDatasetReclustering(
        distance_matrix_dda,
        perturbations_per_vector=3,
        perturbation_scale=0.001,
        sample_size=20,
        target_unique_vectors=1,
        target_unique_vectors_threshold="tight",
        binary_search_steps=30,
    )
    tolerance = tc.calculate_tolerance()
    tuning = distance_matrix_dda.tuning_copy(tc.tuning_array)
    tuning.tolerance = tolerance
    assert tuning.get_dataset_unique_structures() == 1
