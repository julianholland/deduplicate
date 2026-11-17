import numba
import pytest

from deduplicate_lib.core import resources


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1024, 1024),
        (1.5e3, 1500),
        ("512", 512),
        ("512B", 512),
        ("2KB", 2000),
        ("1.5 MB", 1_500_000),
        ("8GB", 8 * 1000**3),
        ("8gib", 8 * 1024**3),
        ("1TiB", 1024**4),
    ],
)
def test_parse_memory_limit(value, expected):
    assert resources.parse_memory_limit(value) == expected


@pytest.mark.parametrize("value", ["auto", "AUTO", None])
def test_parse_memory_limit_auto(value, monkeypatch):
    monkeypatch.setattr(resources, "available_memory", lambda: 10_000)
    assert resources.parse_memory_limit(value) == 5_000


def test_parse_memory_limit_auto_fallback(monkeypatch):
    monkeypatch.setattr(resources, "available_memory", lambda: None)
    assert resources.parse_memory_limit("auto") == resources.FALLBACK_MEMORY_LIMIT


def test_available_memory_is_positive_or_unknown():
    available = resources.available_memory()
    assert available is None or available > 0


@pytest.mark.parametrize("value", ["lots", "8XB", "GB", "-1GB"])
def test_parse_memory_limit_bad_strings(value):
    with pytest.raises(ValueError):
        resources.parse_memory_limit(value)


@pytest.mark.parametrize("value", [0, -5])
def test_parse_memory_limit_must_be_positive(value):
    with pytest.raises(ValueError, match="positive"):
        resources.parse_memory_limit(value)


@pytest.mark.parametrize("value", [True, [1], object()])
def test_parse_memory_limit_bad_types(value):
    with pytest.raises(TypeError):
        resources.parse_memory_limit(value)


def test_chunk_rows():
    # a quarter of the limit, 8 bytes per value
    assert resources.chunk_rows(dim=128, memory_limit=4 * 8 * 128 * 100) == 100
    assert resources.chunk_rows(dim=128, memory_limit=1) == 1


def test_perturbation_batch():
    # half of the limit, 8 bytes per vector per perturbation
    assert resources.perturbation_batch(1000, 200, 2 * 8 * 1000 * 7) == 7
    assert resources.perturbation_batch(1000, 5, 10**12) == 5
    assert resources.perturbation_batch(10**9, 200, 1) == 1


def test_set_threads():
    before = numba.get_num_threads()
    try:
        resources.set_threads(1)
        assert numba.get_num_threads() == 1
        resources.set_threads(None)
        assert numba.get_num_threads() == 1
        resources.set_threads(10**6)  # capped at numba's maximum
        assert numba.get_num_threads() == numba.config.NUMBA_NUM_THREADS
    finally:
        numba.set_num_threads(before)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            resources.set_threads(bad)


def test_progress_bar_disabled_passes_through():
    assert list(resources.progress_bar(range(3), enabled=False)) == [0, 1, 2]


def test_progress_bar_enabled(capsys):
    assert list(resources.progress_bar(range(3), enabled=True, total=3)) == [0, 1, 2]


def test_progress_bar_without_tqdm_warns(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("tqdm"):
            raise ImportError("no tqdm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.warns(UserWarning, match=r"deduplicate_lib\[progress\]"):
        assert list(resources.progress_bar(range(2), enabled=True)) == [0, 1]


def test_meminfo_available_parses_memavailable(tmp_path):
    fake = tmp_path / "meminfo"
    fake.write_text(
        "MemTotal:       16000000 kB\n"
        "MemFree:         1000000 kB\n"
        "MemAvailable:    8000000 kB\n"
        "Buffers:          100000 kB\n"
    )
    assert resources._meminfo_available(str(fake)) == 8000000 * 1024


def test_meminfo_available_missing_file(tmp_path):
    assert resources._meminfo_available(str(tmp_path / "nope")) is None


def test_meminfo_available_without_memavailable_line(tmp_path):
    fake = tmp_path / "meminfo"
    fake.write_text("MemTotal: 1 kB\n")
    assert resources._meminfo_available(str(fake)) is None


def test_meminfo_available_malformed_value(tmp_path):
    fake = tmp_path / "meminfo"
    fake.write_text("MemAvailable: lots kB\n")
    assert resources._meminfo_available(str(fake)) is None


def test_available_memory_falls_back_when_meminfo_missing(monkeypatch):
    monkeypatch.setattr(resources, "_meminfo_available", lambda *a, **k: None)
    value = resources.available_memory()
    assert value is None or value > 0
