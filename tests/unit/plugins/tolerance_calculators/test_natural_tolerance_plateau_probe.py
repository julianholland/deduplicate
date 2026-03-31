import pytest
import numpy as np
from deduplicate_lib.plugins.tolerance_calculators.natural_tolerance_plateau_probe import (
    NaturalTolerancePlateauProbe,
)


def test_str_representation(distance_matrix_dda):
    assert (
        str(
            NaturalTolerancePlateauProbe(
                duplicate_detection_algorithm_object=distance_matrix_dda,
                perturbations_per_vector=1000,
                perturbation_scale=0.1,
            )
        )
        == f"NaturalTolerancePlateauProbe(perturbations_per_vector=1000, perturbation_scale=0.1, dda={str(distance_matrix_dda).split('(')[0]})"
    )


def test_tolerance_probe(distance_matrix_dda):
    tc = NaturalTolerancePlateauProbe(
        duplicate_detection_algorithm_object=distance_matrix_dda,
        perturbations_per_vector=100,
        perturbation_scale=0.01,
    )
    tc.create_perturbed_dataset(seed=803)
    tc.duplicate_detection_algorithm_object.pre_dda_processing(
        tc.tolerance_dataset_array
    )

    tolerance_results = tc.tolerance_probe(
        lower_tolerance=0.0, upper_tolerance=tc.perturbation_scale, tolerance_steps=10
    )
    assert isinstance(tolerance_results, dict)
    assert len(tolerance_results) == 10
    for tol, unique_structures in tolerance_results.items():
        assert isinstance(tol, float)
        print(type(unique_structures))
        assert isinstance(unique_structures, int | np.integer)
        assert unique_structures > 0


def test_find_plateaus(distance_matrix_dda):
    tc = NaturalTolerancePlateauProbe(
        duplicate_detection_algorithm_object=distance_matrix_dda,
        perturbations_per_vector=100,
        perturbation_scale=0.01,
    )
    number_of_datapoints = 100
    tol_values = np.linspace(0.0, tc.perturbation_scale, number_of_datapoints)
    three_plateau_unique_vectors = (
        [1] * (number_of_datapoints // 3)
        + [2] * (number_of_datapoints // 3)
        + [3] * (number_of_datapoints // 3)
        + [4] * (number_of_datapoints - 2 * (number_of_datapoints // 3))
    )
    three_plateau_results = {
        tol: unique_vectors
        for tol, unique_vectors in zip(tol_values, three_plateau_unique_vectors)
    }
    no_plateau_results = {tol: i for i, tol in enumerate(tol_values)}

    plateaus = tc.find_plateaus(
        three_plateau_results, datapoints_to_calculate_gradient=2, plateau_threshold=0.1
    )
    assert len(plateaus) == 3

    for start_tol, end_tol, length in plateaus:
        print(f"Plateau from {start_tol} to {end_tol} with length {length}")
        assert end_tol > start_tol
        assert length >= 2

    no_plateaus = tc.find_plateaus(
        no_plateau_results, datapoints_to_calculate_gradient=2, plateau_threshold=0.1
    )
    assert len(no_plateaus) == 0


@pytest.mark.parametrize("dda_fixture", ["distance_matrix_dda", "multi_hashing_dda"])
def test_calculate_tolerance(request, dda_fixture):
    dda = request.getfixturevalue(dda_fixture)

    tc = NaturalTolerancePlateauProbe(
        duplicate_detection_algorithm_object=dda,
        perturbations_per_vector=1,
        perturbation_scale=0.001,
        probe_steps=100,
    )

    x_dupes = 33
    rng = np.random.default_rng(seed=803)
    low_dataset=rng.uniform(low=0.0, high=0.3, size=(int(100-x_dupes/2), 200))
    high_dataset=rng.uniform(low=0.7, high=1.0, size=(int(100-x_dupes/2), 200))
    mid_dataset=rng.normal(loc=0.5, scale=0.001, size=(x_dupes, 200))
    tc.duplicate_detection_algorithm_object.dataset_array=np.vstack([low_dataset, high_dataset, mid_dataset])
    # tc.tolerance_dataset_array=

    # generate rattled data and compute distance matrix
    tc.create_perturbed_dataset()
    with tc.temp_attr(
        tc.duplicate_detection_algorithm_object,
        "dataset_array",
        tc.tolerance_dataset_array,
    ):
        tc.duplicate_detection_algorithm_object.pre_dda_processing(
            tc.tolerance_dataset_array
        )

    # tc.tolerance_probe()
    # use these to calculate the tolerance
    # tc.binary_search_steps=
    tolerance = tc.calculate_tolerance()

    assert isinstance(tolerance, float)
    assert tolerance > 0.0
    tc.duplicate_detection_algorithm_object.tolerance = tolerance
    with tc.temp_attr(
        tc.duplicate_detection_algorithm_object,
        "dataset_array",
        tc.tolerance_dataset_array,
    ):
        unique_vectors = (
            tc.duplicate_detection_algorithm_object.get_dataset_unique_structures()
        )
    assert unique_vectors > 0 and unique_vectors <= len(tc.tolerance_dataset_array)
    
    

    tc.probe_steps = 1
    with pytest.raises(ValueError,
                       match="Not enough tolerance steps to calculate gradient with the given datapoints_to_calculate_gradient."):
        tc.calculate_tolerance()

    tc.probe_steps = 5
    with pytest.warns(
        UserWarning,
        match="No plateaus found in tolerance probe. Consider adding in perturbed structures and/or increasing dataset size and/or increaseing probe steps.\nReturning average of all same and all different tolerance as fallback.",
    ):
         tc.calculate_tolerance()

    # assert False

@pytest.mark.parametrize("dda_fixture", ["distance_matrix_dda", "multi_hashing_dda"])
def test_get_plateau_log(request, dda_fixture):
    dda = request.getfixturevalue(dda_fixture)

    tc = NaturalTolerancePlateauProbe(
        duplicate_detection_algorithm_object=dda,
        perturbations_per_vector=100,
        perturbation_scale=0.01,
    )

    number_of_datapoints = 100
    tol_values = np.linspace(0.0, tc.perturbation_scale, number_of_datapoints)
    three_plateau_unique_vectors = (
        [1] * (number_of_datapoints // 3)
        + [2] * (number_of_datapoints // 3)
        + [3] * (number_of_datapoints // 3)
        + [4] * (number_of_datapoints - 2 * (number_of_datapoints // 3))
    )
    three_plateau_results = {
        tol: unique_vectors
        for tol, unique_vectors in zip(tol_values, three_plateau_unique_vectors)
    }

    sorted_tols=sorted(three_plateau_results.keys())
    with pytest.raises(ValueError, match="datapoints_to_calculate_gradient must be greater than 1."):
        tc.get_plateau_log(
            sorted_tols=sorted_tols,
            tolerance_results=three_plateau_results,
            datapoints_to_calculate_gradient=1,
            plateau_threshold=0.1,
        )

    plateau_log = tc.get_plateau_log(sorted_tols, three_plateau_results, datapoints_to_calculate_gradient=2, plateau_threshold=0.1)
        
    
    print(f'plateau log size{len(plateau_log)}\n',
          f'plateau_log: {plateau_log}')

    assert isinstance(plateau_log, np.ndarray)
    assert plateau_log.dtype == bool
    assert len(plateau_log) == number_of_datapoints - 2