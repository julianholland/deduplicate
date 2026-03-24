import pytest
import numpy as np
from deduplicate.core.tolerance_calculator import ToleranceCalculator


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        ToleranceCalculator()


def test_calculate_tolerance(dummy_tolerance_calculator):
    tolerance = dummy_tolerance_calculator.calculate_tolerance()
    assert tolerance == 0.1


def test_create_perturbed_dataset(dummy_tolerance_calculator):
    dummy_tolerance_calculator.perturbations_per_vector = 3000

    def run_per_seed(seed):
        dummy_tolerance_calculator.create_perturbed_dataset(seed=seed)
        assert (
            dummy_tolerance_calculator.tolerance_dataset_array.shape[0]
            == dummy_tolerance_calculator.duplicate_detection_algorithm_object.dataset_array.shape[
                0
            ]
            * dummy_tolerance_calculator.perturbations_per_vector
        )
        assert (
            dummy_tolerance_calculator.tolerance_dataset_array.shape[1]
            == dummy_tolerance_calculator.duplicate_detection_algorithm_object.dataset_array.shape[
                1
            ]
        )

        std_dev_of_perturbations = np.std(
            dummy_tolerance_calculator.tolerance_dataset_array
            - np.repeat(
                dummy_tolerance_calculator.duplicate_detection_algorithm_object.dataset_array,
                dummy_tolerance_calculator.perturbations_per_vector,
                axis=0,
            ),
            axis=0,
        )

        assert np.allclose(
            std_dev_of_perturbations,
            dummy_tolerance_calculator.perturbation_scale,
            atol=0.05 * dummy_tolerance_calculator.perturbation_scale,
        ), f" {std_dev_of_perturbations} too different for seed {seed}"

        original = dummy_tolerance_calculator.duplicate_detection_algorithm_object.dataset_array
        perturbed = dummy_tolerance_calculator.tolerance_dataset_array
        for i in range(original.shape[0]):
            assert np.allclose(
                perturbed[i * dummy_tolerance_calculator.perturbations_per_vector],
                original[i],
            )

    for seed in list(range(803, 1000)):
        run_per_seed(seed)


def test_ensure_perturbed_dataset(dummy_tolerance_calculator):

    dummy_tolerance_calculator.create_perturbed_dataset(seed=803)
    dummy_tolerance_calculator._ensure_perturbed_dataset()
    correct_shape = (
        dummy_tolerance_calculator.duplicate_detection_algorithm_object.dataset_array.shape[
            0
        ]
        * dummy_tolerance_calculator.perturbations_per_vector,
        dummy_tolerance_calculator.duplicate_detection_algorithm_object.dataset_array.shape[
            1
        ],
    )
    assert dummy_tolerance_calculator.tolerance_dataset_array.shape == correct_shape

    dummy_tolerance_calculator.tolerance_dataset_array = np.array(
        []
    )  # Reset to empty to force regeneration
    with pytest.warns(
        UserWarning,
        match="Perturbed dataset is not properly initialized. Recreating it.\nEnsure a perturbed dataset is set prior finding tolerance.",
    ):
        dummy_tolerance_calculator._ensure_perturbed_dataset()
    assert dummy_tolerance_calculator.tolerance_dataset_array.shape == correct_shape


def test_temp_attr(dummy_tolerance_calculator):
    original_value = (
        dummy_tolerance_calculator.duplicate_detection_algorithm_object.tolerance
    )
    with dummy_tolerance_calculator.temp_attr(
        dummy_tolerance_calculator.duplicate_detection_algorithm_object,
        "tolerance",
        0.5,
    ):
        assert (
            dummy_tolerance_calculator.duplicate_detection_algorithm_object.tolerance
            == 0.5
        )
    assert (
        dummy_tolerance_calculator.duplicate_detection_algorithm_object.tolerance
        == original_value
    )


def test_binary_search_tolerance(dummy_tolerance_calculator):

    def run_per_seed(seed: int, target: int):
        dummy_tolerance_calculator.create_perturbed_dataset(seed=seed)
        tolerance = dummy_tolerance_calculator.binary_search_tolerance(
            target_unique_vectors=1, find_largest_tolerance_for_target=True
        )
        assert np.isclose(tolerance, target, atol=0.05)

    dummy_tolerance_calculator.perturbation_scale = 0.001
    dummy_tolerance_calculator.perturbations_per_vector = 5

    dummy_tolerance_calculator.binary_search_steps = 0
    dummy_tolerance_calculator.create_perturbed_dataset(seed=803)
    expected_tolerance_no_search = (
        np.ptp(dummy_tolerance_calculator.tolerance_dataset_array) / 2
    )
    with pytest.warns(
        UserWarning,
        match=f"Binary search called with {dummy_tolerance_calculator.binary_search_steps} steps. No binary search performed.\n Returning tolerance of half the range of the perturbed dataset: {expected_tolerance_no_search}.",
    ):
        assert (
            dummy_tolerance_calculator.binary_search_tolerance(
                target_unique_vectors=1, find_largest_tolerance_for_target=True
            )
            == expected_tolerance_no_search
        )

    # one step binary search should be half the range of values in the perturbed dataset, which should be around 1.0 for the default perturbation scale of and then one step of the binary search should give us a tolerance of around 0.5
    dummy_tolerance_calculator.binary_search_steps = 1
    for seed in list(range(803, 1000)):
        run_per_seed(seed, 0.5)

    # tolerance when find largest target will converge to half the max perturbed value, which is around 1.0
    dummy_tolerance_calculator.binary_search_steps = 10
    for seed in list(range(803, 1000)):
        run_per_seed(seed, 1.0)
