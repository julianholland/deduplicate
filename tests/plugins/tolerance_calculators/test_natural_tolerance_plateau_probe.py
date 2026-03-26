import pytest
import numpy as np
from deduplicate.plugins.tolerance_calculators.natural_tolerance_plateau_probe import (
    NaturalTolerancePlateauProbe,
)
from deduplicate.plugins.duplicate_detection_algorithms.distance_matrix import (
    DistanceMatrix,
)
from deduplicate.plugins.duplicate_detection_algorithms.multi_hashing import (
    MultiHashing,
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
        assert isinstance(unique_structures, int)
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
    three_plateau_results = {tol: unique_vectors for tol, unique_vectors in zip(tol_values, three_plateau_unique_vectors)}
    no_plateau_results = {tol: i for i, tol in enumerate(tol_values)}

    plateaus = tc.find_plateaus(three_plateau_results, datapoints_to_calculate_gradient=2, plateau_threshold=0.1)
    assert len(plateaus) == 3
    
    for start_tol, end_tol, length in plateaus:
        print(f"Plateau from {start_tol} to {end_tol} with length {length}")
        assert end_tol > start_tol
        assert length >= 2
    
    no_plateaus = tc.find_plateaus(no_plateau_results, datapoints_to_calculate_gradient=2, plateau_threshold=0.1)
    assert len(no_plateaus) == 0


@pytest.mark.parametrize(
    "dda_fixture",
    ["distance_matrix_dda", "multi_hashing_dda"]
)
def test_calculate_tolerance(request, dda_fixture):
    dda = request.getfixturevalue(dda_fixture)

    tc = NaturalTolerancePlateauProbe(
        duplicate_detection_algorithm_object=dda,
        perturbations_per_vector=100,
        perturbation_scale=0.01,
    )

    # generate rattled data and compute distance matrix
    tc.create_perturbed_dataset()
    with tc.temp_attr(
        tc.duplicate_detection_algorithm_object,
        "dataset_array",
        tc.tolerance_dataset_array,
    ):
        tc.duplicate_detection_algorithm_object.pre_dda_processing(tc.tolerance_dataset_array)

    # use these to calculate the tolerance
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
    with pytest.warns(
        UserWarning,
        match="No plateaus found in tolerance probe. Consider adding in perturbed structures and/or increasing dataset size.\nReturning average of all same and all different tolerance as fallback."
    ):
        tolerance = tc.calculate_tolerance()
