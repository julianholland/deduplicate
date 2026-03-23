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
                original[i]
            )
    
    for seed in list(range(803, 1000)):
        run_per_seed(seed)
# def test_ensure_perturbed_dataset(dummy_tolerance_calculator):


# def test_binary_search_tolerance(dummy_tolerance_calculator):
#     dummy_tolerance_calculator
