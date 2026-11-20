"""Monte Carlo check: sampling the surrogate converges to TSEUQLib's values.

TSEUQLib integrates the Taylor series T_n exactly. Sampling T_n itself (not
the model) must then approach the library's numbers with error shrinking
like N^(-1/2) and no floor: a mistake in the library would show as the error
levelling off at its size. Here T_3 of the cantilever in
examples/cantilever_beam.py (five inputs, one of each distribution), up to
10^7 samples, median of 10 independent runs. Writes
examples/plots/mc_convergence.png.

Run from the repository root: python -B examples/verification/mc_convergence.py
(--quick: up to 10^5 samples, 3 runs)
"""

import argparse
import os
import sys
from itertools import combinations_with_replacement
from math import factorial

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))  # examples/, for cantilever_beam

import numpy as np  # noqa: E402

import cantilever_beam as cb  # noqa: E402

OUT = os.path.join(HERE, "..", "plots", "mc_convergence.png")
N_ORDER = 3
QTYS = ["mean", "variance", "skewness", "kurtosis"]


def coefficients(y, d, n):
    """{alpha: c_alpha} of T_n, in powers of x - mean."""
    c = {(0,) * d: y.real}
    for k in range(1, n + 1):
        for dirs in combinations_with_replacement(range(1, d + 1), k):
            alpha = tuple(dirs.count(i) for i in range(1, d + 1))
            c[alpha] = y.get_deriv(list(dirs)) / np.prod([factorial(a) for a in alpha])
    return c


def sample_inputs(rng, m):
    """m draws of (P, E, L, w, u) from the distributions of cantilever_beam."""
    A, B = cb.A, cb.B
    v = rng.random(m)
    return np.column_stack([
        rng.normal(A[0], B[0], m),
        rng.lognormal(A[1], B[1], m),
        rng.uniform(A[2], B[2], m),
        rng.triangular(A[3], (A[3] + B[3]) / 2, B[3], m),
        (1 - (1 - v)**(1 / B[4]))**(1 / A[4]),  # Kumaraswamy by inverse CDF
    ])


def evaluate(coeffs, mean, X, n):
    Z = X - mean
    powers = [[np.ones(len(Z))] for _ in range(Z.shape[1])]
    for i in range(Z.shape[1]):
        for _ in range(n):
            powers[i].append(powers[i][-1] * Z[:, i])
    out = np.zeros(len(Z))
    for alpha, c in coeffs.items():
        term = np.full(len(Z), c)
        for i, k in enumerate(alpha):
            if k:
                term *= powers[i][k]
        out += term
    return out


def run(seed, coeffs, mean, checkpoints, shift, chunk=10**6):
    """Sample estimates of the four quantities at each checkpoint N, from
    running power sums of y - shift (shift: the library's mean)."""
    rng = np.random.default_rng(seed)
    sums = np.zeros(5)
    done, out = 0, []
    for N in checkpoints:
        while done < N:
            m = min(chunk, N - done)
            y = evaluate(coeffs, mean, sample_inputs(rng, m), N_ORDER) - shift
            sums += [m, y.sum(), (y**2).sum(), (y**3).sum(), (y**4).sum()]
            done += m
        r1, r2, r3, r4 = sums[1:] / sums[0]
        mu2 = r2 - r1**2
        mu3 = r3 - 3 * r1 * r2 + 2 * r1**3
        mu4 = r4 - 4 * r1 * r3 + 6 * r1**2 * r2 - 3 * r1**4
        out.append({"mean": shift + r1, "variance": mu2, "skewness": mu3 / mu2**1.5, "kurtosis": mu4 / mu2**2})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="up to 1e5 samples, 3 runs")
    args = ap.parse_args()
    top, runs = (5, 3) if args.quick else (7, 10)
    checkpoints = [int(round(10**e)) for e in np.arange(2, top + 0.01, 0.5)]

    tq, y = cb.surrogate(N_ORDER)
    mom = cb.moments(tq, y)
    exact = {"mean": mom["mean"], "variance": mom["sd"]**2, "skewness": mom["skewness"], "kurtosis": mom["kurtosis"]}
    means = surrogate_center()  # the expansion point of T_n
    coeffs = coefficients(y, len(cb.NAMES), N_ORDER)
    errs = {q: np.zeros((runs, len(checkpoints))) for q in QTYS}
    for r in range(runs):
        est = run(1000 + r, coeffs, np.array(means), checkpoints, exact["mean"])
        for j, e in enumerate(est):
            for q in QTYS:
                errs[q][r, j] = abs(e[q] - exact[q]) / abs(exact[q])
        print(f"run {r + 1}/{runs}: relative error at N = {checkpoints[-1]:.0e}: " +
              ", ".join(f"{q} {errs[q][r, -1]:.1e}" for q in QTYS), flush=True)
    plot(checkpoints, errs, runs)


def surrogate_center():
    mom = cb.tuq.rv_central_moments([cb.DISTS, np.array(cb.A, dtype=object), np.array(cb.B, dtype=object)], 2)
    mom.compute_central_moments()
    return [float(np.real(m)) for m in mom.rv_mean]


def plot(checkpoints, errs, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    GRID, AXIS, BLUE, BAND = "#e1e0d9", "#c3c2b7", "#2a78d6", "#cde2fb"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), sharex=True)
    N = np.array(checkpoints, dtype=float)
    for ax, q in zip(axes, QTYS):
        e = errs[q]
        med = np.median(e, axis=0)
        ax.fill_between(N, e.min(axis=0), e.max(axis=0), color=BAND, linewidth=0, zorder=1)
        ax.plot(N, med, color=BLUE, linewidth=2, marker="o", markersize=4, zorder=3)
        ref = med[0] * np.sqrt(N[0] / N)
        ax.plot(N, ref, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(q, loc="left", color=INK, fontsize=10)
        ax.set_xlabel("samples N")
        ax.grid(color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("relative error vs TSEUQLib")
    axes[0].annotate("∝ N$^{-1/2}$", (N[1], (errs["mean"].min(axis=0)[0] * np.sqrt(N[0] / N[1]))),
                     xytext=(0, -14), textcoords="offset points", color=MUTED, fontsize=8)
    fig.suptitle("Sampling the surrogate converges to TSEUQLib's exact values at the Monte Carlo rate",
                 x=0.012, ha="left", fontsize=12.5, color=INK, y=0.98)
    fig.text(0.012, 0.865,
             f"Monte Carlo on T$_{N_ORDER}$ of the cantilever in examples/cantilever_beam.py (five inputs, one of "
             f"each distribution). Line: median of {runs} independent runs; band: their range; dashed: N$^{{-1/2}}$.\n"
             "The error keeps falling with no floor, so TSEUQLib's numbers are what sampling converges to; "
             "a mistake in them would show as a floor.",
             fontsize=9, color=INK2, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
