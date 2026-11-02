"""Draw examples/plots/polynomial_exactness.png from polynomial_results.csv
(written by polynomial_exactness.py).

Relative error = |library - exact| / |exact|; an exactly zero quantity (a
Sobol' index of an interaction the polynomial lacks) is measured on the
scale of its group (same polynomial, order and quantity family). Errors
below 1e-17, exact zeros included, sit at the "exact" floor.
"""

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "polynomial_results.csv")
OUT = os.path.join(os.path.dirname(HERE), "plots", "polynomial_exactness.png")

FAMILIES = ["Expected value", "Variance", "Third central moment", "Fourth central moment",
            "Sobol' indices", "Shapley effects"]
FLOOR, EPS, TOP = 1e-17, np.finfo(float).eps, 1e4

# the reference palette of the dataviz skill (light mode)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BLUE = "#e1e0d9", "#c3c2b7", "#2a78d6"


def load():
    with open(CSV) as fh:
        rows = list(csv.DictReader(fh))
    group = lambda r: (r["distribution"], r["inputs"], r["degree"], r["order"], r["family"])  # noqa: E731
    scale = defaultdict(float)
    for r in rows:
        r["library"], r["exact"] = float(r["library"]), float(r["exact"])
        r["offset"] = int(r["order"]) - int(r["degree"])
        scale[group(r)] = max(scale[group(r)], abs(r["exact"]))
    for r in rows:
        s = scale[group(r)]
        denom = abs(r["exact"]) if abs(r["exact"]) > 1e-25 * s else s
        err = abs(r["library"] - r["exact"]) / denom if denom > 0 else abs(r["library"])
        r["err"] = min(max(err, FLOOR), TOP)
    return rows


def main():
    rows = load()
    offsets = sorted({r["offset"] for r in rows})
    polys = {(r["distribution"], r["inputs"], r["degree"]) for r in rows}
    exact_side = [r for r in rows if r["offset"] >= 0]
    worst = max(r["err"] for r in exact_side)
    degrees = sorted({int(r["degree"]) for r in rows})

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": MUTED,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for ax, fam in zip(axes.flat, FAMILIES):
        pts = [r for r in rows if r["family"] == fam]
        x = np.array([r["offset"] for r in pts]) + rng.uniform(-0.28, 0.28, len(pts))
        y = [r["err"] for r in pts]
        ax.scatter(x, y, s=12, color=BLUE, edgecolor=SURFACE, linewidth=0.4, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(FLOOR / 4, TOP * 4)
        ax.set_yticks([FLOOR, 1e-15, 1e-10, 1e-5, 1e0])
        ax.set_yticklabels(["exact", r"$10^{-15}$", r"$10^{-10}$", r"$10^{-5}$", r"$10^{0}$"])
        ax.minorticks_off()
        ax.axhline(EPS, color=AXIS, linewidth=1, zorder=1)
        ax.axvline(-0.5, color=AXIS, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
        ax.set_xticks(offsets)
        ax.set_xticklabels([("p" if o == 0 else f"p{o:+d}").replace("-", "−") for o in offsets])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        n_fam = len(pts)
        ax.set_title(f"{fam}\n{n_fam} checks", fontsize=9.5, color=INK, loc="left")
    for ax in axes[1]:
        ax.set_xlabel("expansion order n", color=INK2)
    fig.supylabel("relative error vs the model's exact value", color=INK2, fontsize=9, x=0.012)
    # reference-line and region labels, once
    a0 = axes[0, 0]
    # left end of the line: the dots there sit near 1, clear of it
    a0.text(offsets[0] - 0.4, EPS * 2.5, "machine ε", color=MUTED, fontsize=8, ha="left", va="bottom")
    a0.text(-0.62, TOP * 2, "order below degree", color=MUTED, fontsize=8, ha="right", va="top")
    a0.text(-0.38, TOP * 2, "order ≥ degree", color=MUTED, fontsize=8, ha="left", va="top")

    fig.suptitle("For a polynomial model, every quantity is exact once the expansion order reaches its degree",
                 x=0.012, ha="left", fontsize=12.5, color=INK, y=0.985)
    fig.text(0.012, 0.915,
             f"{len(rows)} checks on {len(polys)} random polynomials of degree p = {degrees[0]}-{degrees[-1]} "
             f"(2 or 3 inputs; normal, uniform, triangular, Kumaraswamy and lognormal inputs; every\n"
             f"monomial has a random coefficient), against the model's own exact values (SymPy, 50 digits). "
             f"Worst relative error at n ≥ p: {worst:.0e}.",
             fontsize=9, color=INK2, ha="left", va="top")
    fig.tight_layout(rect=(0.02, 0, 1, 0.86))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
