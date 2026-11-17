"""How large must NaturalTolerancePlateauProbe's gradient window be?

Runs the NTPP sweep on synthetic datasets with a known number of natural
clustering scales, for several dimensions, dataset sizes, probe densities and
both DDAs,
then scores every gradient window (as a share of the sweep) against the known
answer.  Produces the figures in ``docs/_static/ntpp/`` used by the
tolerance-calculator docs.

Usage::

    uv run python docs/scripts/ntpp_window_study.py          # run (cached) + plot
    uv run python docs/scripts/ntpp_window_study.py --plot   # re-plot from the cache

The full run (720 sweeps) takes about 1.5 hours; probe results are cached in
``ntpp_window_study_results.json`` next to this script (not committed), so an
interrupted run resumes and ``--plot`` redraws without recomputing.

Datasets (d-dimensional, d in ``DIMS``):

* ``bimodal``: n/4 groups of 4 near-duplicates (sigma 0.001) at random centres
  in [0, 10]^d.  One natural scale: a plateau at n/4 unique vectors.
* ``trimodal``: S super-clusters (centres in [0, 10]^d), each holding S
  clusters (sigma 0.05), each holding 4 near-duplicates (sigma 0.001), with
  S = sqrt(n/4).  Two natural scales: plateaus at S*S and S.
* ``linear``: n points spread uniformly at random along a line.
* ``uniform``: n points uniformly at random in [0, 10]^d.
* ``unimodal``: n points from one isotropic Gaussian (sigma 1).

The last three have no natural scale: the right answer is no non-trivial
plateau.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from deduplicate_lib.plugins.duplicate_detection_algorithms.distance_matrix import (
    DistanceMatrix,
)
from deduplicate_lib.plugins.duplicate_detection_algorithms.multi_hashing import (
    MultiHashing,
)
from deduplicate_lib.plugins.tolerance_calculators.natural_tolerance_plateau_probe import (
    DEFAULT_WINDOW_FRACTION,
    NaturalTolerancePlateauProbe,
)

DOCS = Path(__file__).resolve().parents[1]
OUT = DOCS / "_static" / "ntpp"
CACHE = Path(__file__).with_name("ntpp_window_study_results.json")

# dimension -> probe densities run at it (the full set only at the lowest
# dimension, which is enough to show the window scales with probe_steps)
DIMS = {8: (50, 100, 200, 400), 64: (100, 200), 256: (100, 200)}
SIZES = (100, 400, 1600)
SEEDS = (1, 2, 3)
DDAS = {"DistanceMatrix": DistanceMatrix, "MultiHashing": MultiHashing}
WINDOW_FRACTIONS = np.round(np.arange(0.02, 0.201, 0.01), 2)
CHOSEN_FRACTION = DEFAULT_WINDOW_FRACTION


def bimodal(n, dim, rng):
    centres = rng.uniform(0, 10, (n // 4, dim))
    data = np.repeat(centres, 4, axis=0) + rng.normal(0, 0.001, (n // 4 * 4, dim))
    return data, [n // 4]


def trimodal(n, dim, rng):
    s = round(np.sqrt(n / 4))
    supers = rng.uniform(0, 10, (s, dim))
    clusters = np.repeat(supers, s, axis=0) + rng.normal(0, 0.05, (s * s, dim))
    data = np.repeat(clusters, 4, axis=0) + rng.normal(0, 0.001, (s * s * 4, dim))
    return data, [s * s, s]


def linear(n, dim, rng):
    direction = rng.normal(size=dim)
    direction /= np.linalg.norm(direction)
    positions = np.sort(rng.uniform(0, 10, n))
    return positions[:, None] * direction[None, :], []


def uniform(n, dim, rng):
    return rng.uniform(0, 10, (n, dim)), []


def unimodal(n, dim, rng):
    return rng.normal(0, 1, (n, dim)), []


DATASETS = {
    "bimodal": bimodal,
    "trimodal": trimodal,
    "linear": linear,
    "uniform": uniform,
    "unimodal": unimodal,
}


def window_for(fraction, steps):
    return max(2, round(fraction * steps))


def non_trivial_counts(tc, probe_results, window):
    tc.datapoints_to_calculate_gradient = window
    plateaus = tc.find_plateaus(probe_results)
    return sorted((p.unique_vectors for p in plateaus if not p.trivial), reverse=True)


def is_correct(counts, truth):
    if len(counts) != len(truth):
        return False
    return all(abs(c - t) <= max(1, 0.05 * t) for c, t in zip(counts, truth))


def run_study():
    """Run every sweep once; cache the probe results."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    configs = [
        (name, dim, n, dda_name, steps, seed)
        for dim, probe_steps in DIMS.items()
        for name in DATASETS
        for n in SIZES
        for dda_name in DDAS
        for steps in probe_steps
        for seed in SEEDS
    ]
    for name, dim, n, dda_name, steps, seed in configs:
        key = f"{name}|{dim}|{n}|{dda_name}|{steps}|{seed}"
        if key in cache:
            continue
        data, truth = DATASETS[name](n, dim, np.random.default_rng(seed))
        dda = DDAS[dda_name](dataset_array=data, max_vector_array_size=n)
        tc = NaturalTolerancePlateauProbe(dda, probe_steps=steps)
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = tc.probe()
        cache[key] = {
            "truth": truth,
            "max_unique": tc.max_unique_vectors,
            "tolerances": list(results),
            "counts": list(results.values()),
        }
        print(f"{key}: {time.perf_counter() - start:.1f}s", flush=True)
        CACHE.write_text(json.dumps(cache))
    return cache


def score(cache):
    """{(dataset, dim, n, dda, steps): one correct/incorrect row over WINDOW_FRACTIONS per seed}."""
    tc = NaturalTolerancePlateauProbe(
        DistanceMatrix(dataset_array=np.zeros((1, 1)), max_vector_array_size=1)
    )
    success = {}
    for key, entry in cache.items():
        name, dim, n, dda_name, steps, _seed = key.split("|")
        steps = int(steps)
        results = dict(zip(entry["tolerances"], entry["counts"]))
        tc.max_unique_vectors = entry["max_unique"]
        row = [
            is_correct(
                non_trivial_counts(tc, results, window_for(f, steps)), entry["truth"]
            )
            for f in WINDOW_FRACTIONS
        ]
        success.setdefault((name, int(dim), int(n), dda_name, steps), []).append(row)
    return success


def share_correct(success, **fixed):
    """Share of runs correct per window, pooled over every key field not fixed."""
    fields = ("name", "dim", "n", "dda", "steps")
    rows = [
        row
        for key, key_rows in success.items()
        if all(key[fields.index(f)] == v for f, v in fixed.items())
        for row in key_rows
    ]
    return np.mean(rows, axis=0), len(rows)


# --- plotting ---------------------------------------------------------------

SURFACE = "#fcfcfb"
TEXT = "#1a1a19"
MUTED = "#6b6a63"
GRID = "#e4e3dd"
SERIES_1 = "#2a78d6"
DDA_COLORS = {"DistanceMatrix": "#2a78d6", "MultiHashing": "#eb6834"}  # slots 1, 2
PLATEAU_FILL = "#cde2fb"  # blue 100
TRIVIAL_FILL = "#e4e3dd"
# ordinal blue ramp for probe_steps 50 -> 400 (light -> dark)
STEPS_COLORS = {50: "#86b6ef", 100: "#3987e5", 200: "#1c5cab", 400: "#0d366b"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_curves(cache, plt):
    """One sweep per dataset with the plateaus NTPP finds at the chosen window."""
    dim, n, dda_name, steps, seed = 8, 400, "DistanceMatrix", 200, 1
    fig, grid = plt.subplots(2, 3, figsize=(11, 6.4), facecolor=SURFACE)
    axes = grid.ravel()
    for ax in axes[len(DATASETS) :]:
        ax.set_visible(False)
    tc = NaturalTolerancePlateauProbe(
        DistanceMatrix(dataset_array=np.zeros((1, 1)), max_vector_array_size=1)
    )
    for ax, name in zip(axes, DATASETS):
        entry = cache[f"{name}|{dim}|{n}|{dda_name}|{steps}|{seed}"]
        results = dict(zip(entry["tolerances"], entry["counts"]))
        tc.max_unique_vectors = entry["max_unique"]
        tc.datapoints_to_calculate_gradient = window_for(CHOSEN_FRACTION, steps)
        style(ax)
        for p in tc.find_plateaus(results):
            ax.axvspan(
                p.start_tolerance,
                p.end_tolerance,
                color=TRIVIAL_FILL if p.trivial else PLATEAU_FILL,
                linewidth=0,
            )
            if not p.trivial:
                ax.annotate(
                    f"{p.unique_vectors}",
                    (p.tolerance, p.unique_vectors),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=8,
                    color=TEXT,
                )
        ax.plot(entry["tolerances"], entry["counts"], color=SERIES_1, linewidth=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.6, max(entry["counts"]) * 4)  # headroom for plateau labels
        truth = entry["truth"]
        expected = ", ".join(str(t) for t in truth) if truth else "none"
        ax.set_title(f"{name} (expected plateaus: {expected})", fontsize=10, color=TEXT)
        ax.set_xlabel("tolerance", fontsize=9, color=MUTED)
    for ax in grid[:, 0]:
        ax.set_ylabel("unique vectors", fontsize=9, color=MUTED)
    fig.text(
        0.5,
        0.005,
        f"d = {dim}, n = {n}, {dda_name}, probe_steps = {steps}, window = "
        f"{CHOSEN_FRACTION:.0%} of the sweep.  Blue: plateaus found; grey: trivial ends.",
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "ntpp_curves.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def finish_panel(ax, title=None, ylabel=None, xlabel=True):
    ax.axvline(CHOSEN_FRACTION * 100, color=MUTED, linewidth=1, linestyle="--")
    ax.set_ylim(-0.05, 1.05)
    if title:
        ax.set_title(title, fontsize=10, color=TEXT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    if xlabel:
        ax.set_xlabel("window (% of sweep)", fontsize=9, color=MUTED)


def legend_and_title(fig, ax, title, ncol):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=ncol,
        frameon=False,
        fontsize=8,
        labelcolor=TEXT,
    )
    fig.suptitle(title, fontsize=10, color=TEXT, y=0.995)


def plot_windows(success, plt):
    """Per dataset: share of runs with exactly the expected plateaus vs window
    size, by dimension (rows) and dataset size (columns), one line per DDA,
    pooled over probe_steps and seeds."""
    x = WINDOW_FRACTIONS * 100
    dims = list(DIMS)
    for name in DATASETS:
        fig, axes = plt.subplots(
            len(dims),
            len(SIZES),
            figsize=(11, 2.5 * len(dims) + 0.8),
            sharex=True,
            sharey=True,
            facecolor=SURFACE,
        )
        for i, dim in enumerate(dims):
            for j, n in enumerate(SIZES):
                ax = axes[i, j]
                style(ax)
                for dda_name in DDAS:
                    share, _ = share_correct(
                        success, name=name, dim=dim, n=n, dda=dda_name
                    )
                    ax.plot(
                        x,
                        share,
                        color=DDA_COLORS[dda_name],
                        linewidth=2,
                        marker="o",
                        markersize=4,
                        label=dda_name,
                    )
                finish_panel(
                    ax,
                    title=f"n = {n}" if i == 0 else None,
                    ylabel=f"d = {dim}\nshare correct" if j == 0 else None,
                    xlabel=i == len(dims) - 1,
                )
        legend_and_title(
            fig,
            axes[0, 0],
            f"{name}: share of runs where NTPP finds exactly the expected plateaus",
            len(DDAS),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(OUT / f"ntpp_window_{name}.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)


def plot_probe_density(success, plt):
    """At the lowest dimension (all probe densities run): share correct vs
    window, one line per probe_steps, pooled over datasets, sizes and seeds."""
    x = WINDOW_FRACTIONS * 100
    dim = min(DIMS)
    fig, axes = plt.subplots(
        1, len(DDAS), figsize=(11, 3.4), sharey=True, facecolor=SURFACE
    )
    for j, dda_name in enumerate(DDAS):
        ax = axes[j]
        style(ax)
        for steps in DIMS[dim]:
            share, _ = share_correct(success, dim=dim, dda=dda_name, steps=steps)
            ax.plot(
                x,
                share,
                color=STEPS_COLORS[steps],
                linewidth=2,
                marker="o",
                markersize=4,
                label=f"probe_steps = {steps}",
            )
        finish_panel(ax, title=dda_name, ylabel="share correct" if j == 0 else None)
    legend_and_title(
        fig,
        axes[0],
        f"All datasets and sizes, d = {dim}: the usable window is a share of the "
        "sweep, whatever probe_steps is",
        len(DIMS[dim]),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(OUT / "ntpp_window_probe_steps.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def summary(success):
    """Print, per configuration, the windows at which every run is correct."""
    for key in sorted(success):
        ok = WINDOW_FRACTIONS[np.all(success[key], axis=0)]
        span = f"{ok.min():.0%}-{ok.max():.0%}" if ok.size else "never"
        print(f"{'|'.join(map(str, key)):45s} all seeds correct at: {span}")
    print()
    overall, runs = share_correct(success)
    for f, share in zip(WINDOW_FRACTIONS, overall):
        print(f"window {f:.0%}: {share:.1%} of {runs} runs correct")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--plot", action="store_true", help="only re-plot from the cache"
    )
    args = parser.parse_args()
    cache = json.loads(CACHE.read_text()) if args.plot else run_study()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    success = score(cache)
    summary(success)
    plot_curves(cache, plt)
    plot_windows(success, plt)
    plot_probe_density(success, plt)


if __name__ == "__main__":
    main()
