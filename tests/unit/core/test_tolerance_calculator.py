import numpy as np
import pytest

from deduplicate_lib.core.tolerance_calculator import (
    ToleranceCalculator,
    draw_sample,
)
from deduplicate_lib.core.vector_store import InMemoryStore
from deduplicate_lib.plugins.duplicate_detection_algorithms.distance_matrix import (
    DistanceMatrix,
)


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        ToleranceCalculator()  # type: ignore


def test_calculate_tolerance(dummy_tolerance_calculator):
    assert dummy_tolerance_calculator.calculate_tolerance() == 0.1


def test_removed_api_is_gone(dummy_tolerance_calculator):
    for name in (
        "tolerance_dataset_array",
        "temp_attr",
        "create_perturbed_dataset",
        "_ensure_perturbed_dataset",
    ):
        assert not hasattr(dummy_tolerance_calculator, name)


def test_invalid_sampling_raises_in_init(dummy_dda):
    from tests.conftest import DummyToleranceCalculator

    with pytest.raises(ValueError, match="Invalid sampling"):
        DummyToleranceCalculator(dummy_dda, sampling="nope")


# --- draw_sample ---

STORE = InMemoryStore(np.arange(500 * 3, dtype=float).reshape(500, 3))


@pytest.mark.parametrize("sampling", ["blocks", "random", "head"])
def test_draw_sample_modes(sampling):
    sample = draw_sample(STORE, 120, sampling=sampling, block_rows=20, seed=1)
    assert sample.shape == (120, 3)
    assert sample.dtype == float
    # every sampled row is a row of the store
    assert np.isin(sample[:, 0], STORE.read(0, 500)[:, 0]).all()
    again = draw_sample(STORE, 120, sampling=sampling, block_rows=20, seed=1)
    assert np.array_equal(sample, again)


def test_draw_sample_head_is_first_rows():
    assert np.array_equal(
        draw_sample(STORE, 10, sampling="head"), STORE.read(0, 10).astype(float)
    )


def test_draw_sample_blocks_are_contiguous():
    sample = draw_sample(STORE, 100, sampling="blocks", block_rows=20, seed=2)
    ids = sample[:, 0] / 3
    assert (ids[::20] % 20 == 0).all()
    assert (np.diff(ids.reshape(5, 20), axis=1) == 1).all()


def test_draw_sample_random_differs_by_seed():
    a = draw_sample(STORE, 50, sampling="random", seed=1)
    b = draw_sample(STORE, 50, sampling="random", seed=2)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("sampling", ["blocks", "random", "head"])
def test_draw_sample_small_store_returns_everything(sampling):
    small = InMemoryStore(np.arange(12.0).reshape(6, 2))
    sample = draw_sample(small, 100, sampling=sampling)
    assert sorted(sample[:, 0]) == list(small.read(0, 6)[:, 0])


def test_draw_sample_invalid_mode():
    with pytest.raises(ValueError, match="Invalid sampling"):
        draw_sample(STORE, 10, sampling="bogus")


@pytest.mark.parametrize("size", [0, -3])
def test_draw_sample_rejects_bad_size(size):
    with pytest.raises(ValueError, match="sample_size"):
        draw_sample(STORE, size)


def test_draw_sample_empty_store():
    with pytest.raises(ValueError, match="empty"):
        draw_sample(InMemoryStore(np.empty((0, 3))), 10)


# --- base-class helpers ---


def test_prepare_tuning_leaves_dda_untouched(dummy_tolerance_calculator):
    tc = dummy_tolerance_calculator
    dda = tc.duplicate_detection_algorithm_object
    before = dda.get_filled_dataset_array().copy()
    tuning = tc._prepare_tuning()
    assert tuning is not dda
    assert type(tuning) is type(dda)
    assert np.array_equal(tc.tuning_array, before)
    assert np.array_equal(dda.get_filled_dataset_array(), before)


def test_tolerance_upper_bound():
    dda = DistanceMatrix(dataset_array=np.array([[0.0, 0.0], [1.0, 1.0]]))
    from tests.conftest import DummyToleranceCalculator

    tc = DummyToleranceCalculator(dda)
    tc._prepare_tuning()
    assert tc.tolerance_upper_bound() > 2.0  # above the largest L1 distance


def test_binary_search_tolerance_finds_target():
    rng = np.random.default_rng(0)
    data = np.repeat(rng.uniform(0, 10, (20, 4)), 3, axis=0) + rng.normal(
        0, 1e-3, (60, 4)
    )
    dda = DistanceMatrix(dataset_array=data)
    from tests.conftest import DummyToleranceCalculator

    tc = DummyToleranceCalculator(dda, binary_search_steps=40)
    tuning = tc._prepare_tuning()
    tol = tc.binary_search_tolerance(tuning, 20, find_largest_tolerance_for_target=True)
    tuning.tolerance = tol
    assert tuning.get_dataset_unique_structures() == 20
    assert dda.tolerance != tol  # user's DDA untouched


def test_binary_search_zero_steps_warns_and_returns_upper_bound(
    dummy_tolerance_calculator,
):
    tc = dummy_tolerance_calculator
    tuning = tc._prepare_tuning()
    tc.binary_search_steps = 0
    with pytest.warns(UserWarning, match="No binary search performed"):
        tol = tc.binary_search_tolerance(tuning, 1)
    assert tol == tc.tolerance_upper_bound()


def test_binary_search_no_exact_target_warns():
    data = np.array([[0.0], [1.0], [10.0]])
    from tests.conftest import DummyToleranceCalculator

    tc = DummyToleranceCalculator(
        DistanceMatrix(dataset_array=data), binary_search_steps=15
    )
    tuning = tc._prepare_tuning()
    with pytest.warns(UserWarning, match="No exact tolerance found"):
        tc.binary_search_tolerance(tuning, 7)
