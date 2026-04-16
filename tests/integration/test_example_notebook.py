from nbclient import NotebookClient
import nbformat
from deduplicate_lib.plugins.duplicate_detection_algorithms.multi_hashing import MultiHashing
from deduplicate_lib.plugins.tolerance_calculators.natural_tolerance_plateau_probe import NaturalTolerancePlateauProbe
import numpy as np


def test_multi_hashing_example_notebook_runs():
    with open("examples/multi_hashing_demo.ipynb") as f:
        nb = nbformat.read(f, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()

def test_multi_hashing_example_notebook_output():
    # full workflow executed in the notebook

    rng = np.random.default_rng(803) # for reproducibility

    # Create a dataset of 20 vectors with 10 dimensions (any dimensionality would work, but 3 is nice for visualization)
    dataset = rng.uniform(0, 1, size=(20, 10))

    # about each vector we create similar vectors by adding small random noise, these represent vectors that we should consider duplicates
    cluster_size=5
    noisy_dataset = np.zeros((len(dataset) * cluster_size, dataset.shape[1])) # preallocate for original + 4 similar vectors
    for i in range(len(dataset)):
        noisy_dataset[i*cluster_size] = dataset[i] # original vector
        for j in range(1, cluster_size):
            noise = rng.normal(0, 0.005, size=(dataset.shape[1])) # small noise
            noisy_dataset[i*cluster_size + j] = dataset[i % len(dataset)] + noise

    dda = MultiHashing(dataset_array=noisy_dataset, perturbations=10)
    dda.pre_dda_processing()

    tc = NaturalTolerancePlateauProbe(duplicate_detection_algorithm_object=dda,
                                    tolerance_dataset_array=noisy_dataset,)

    tolerance = tc.calculate_tolerance()

    dda.tolerance = tolerance
    dda.get_dataset_unique_structures()
    dda.deduplicate()

    unique_indices = dda.get_unique_vector_indices()
    valid_indices = {}
    for i in range(len(dataset)):
        valid_indices[i] = list(unique_indices[unique_indices // cluster_size == i])

    precomputed_valid_indices = {0: [0],
                                1: [5],
                                2: [10],
                                3: [15],
                                4: [20],
                                5: [25, 26],
                                6: [30],
                                7: [35, 36],
                                8: [40],
                                9: [45],
                                10: [50],
                                11: [55],
                                12: [60, 64],
                                13: [65],
                                14: [70],
                                15: [75],
                                16: [80],
                                17: [85],
                                18: [90],
                                19: [95]}

    assert len(valid_indices) > 0
    assert valid_indices == precomputed_valid_indices
