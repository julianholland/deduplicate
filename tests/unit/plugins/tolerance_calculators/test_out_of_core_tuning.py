"""Tolerance calculators on out-of-core DDAs: same answer as in memory, DDA untouched."""

import h5py
import numpy as np
import pytest

from deduplicate_lib.core import hash_index
from deduplicate_lib.plugins.duplicate_detection_algorithms.multi_hashing import (
    MultiHashing,
)
from deduplicate_lib.plugins.tolerance_calculators.natural_tolerance_plateau_probe import (
    NaturalTolerancePlateauProbe,
)
from deduplicate_lib.plugins.tolerance_calculators.perturbed_dataset_reclustering import (
    PerturbedDatasetReclustering,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:No exact tolerance found:UserWarning",
    "ignore:No natural plateaus found:UserWarning",
    "ignore:Sigma acceptance threshold:UserWarning",
)


@pytest.fixture(autouse=True)
def no_fsync(monkeypatch):
    monkeypatch.setattr(hash_index, "FSYNC", False)


def make_data():
    rng = np.random.default_rng(5)
    centres = rng.uniform(0, 10, size=(20, 6))
    data = np.repeat(centres, 5, axis=0) + rng.normal(0, 0.01, size=(100, 6))
    return data


DATA = make_data()
MH_KWARGS = {"tolerance": 0.1, "perturbations": 20, "seed": 4}

CALCULATORS = {
    "pdr": lambda dda: PerturbedDatasetReclustering(
        dda,
        perturbations_per_vector=3,
        perturbation_scale=0.01,
        sample_size=40,
        binary_search_steps=12,
    ),
    "ntpp": lambda dda: NaturalTolerancePlateauProbe(
        dda,
        probe_steps=30,
        sample_size=60,
        block_rows=10,
        binary_search_steps=15,
    ),
}


def out_of_core_dda(kind, tmp_path, **extra):
    if kind == "npy":
        path = tmp_path / "data.npy"
        np.save(path, DATA)
    else:
        path = tmp_path / "data.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("vectors", data=DATA, maxshape=(None, DATA.shape[1]))
    return MultiHashing(dataset=path, **MH_KWARGS, **extra)


@pytest.mark.parametrize("calculator", CALCULATORS)
@pytest.mark.parametrize("kind", ["npy", "hdf5"])
def test_calculator_matches_in_memory_and_leaves_dda_alone(kind, calculator, tmp_path):
    in_memory = MultiHashing(dataset_array=DATA, **MH_KWARGS)
    expected = CALCULATORS[calculator](in_memory).calculate_tolerance()

    dda = out_of_core_dda(kind, tmp_path)
    assert dda.is_out_of_core
    tolerance = CALCULATORS[calculator](dda).calculate_tolerance()

    assert isinstance(tolerance, float)
    assert tolerance > 0
    assert tolerance == expected
    assert dda.is_out_of_core
    assert dda.vector_count == len(DATA)
    assert dda.tolerance == MH_KWARGS["tolerance"]


@pytest.mark.parametrize("calculator", CALCULATORS)
def test_calculator_with_index_dir_does_not_touch_the_index(calculator, tmp_path):
    index_dir = tmp_path / "index"
    dda = out_of_core_dda("npy", tmp_path, index_dir=index_dir)
    dda.get_dataset_unique_structures()
    metadata = (index_dir / "metadata.json").read_bytes()
    files = sorted(p.name for p in index_dir.iterdir())

    expected = CALCULATORS[calculator](
        MultiHashing(dataset_array=DATA, **MH_KWARGS)
    ).calculate_tolerance()
    assert CALCULATORS[calculator](dda).calculate_tolerance() == expected

    assert (index_dir / "metadata.json").read_bytes() == metadata
    assert sorted(p.name for p in index_dir.iterdir()) == files
    assert dda.is_out_of_core
    assert dda.vector_count == len(DATA)
    assert dda.tolerance == MH_KWARGS["tolerance"]
