"""Draw verification_results.csv (from verify_all.py): every check's relative
error against the exact symbolic answer, by distribution and quantity.
Writes examples/plots/verification.png.

Relative error = |library - exact| / max(|exact|, 1e-12 * S), where S is the
largest |exact| in the same group (function, quantity family, value or
sensitivity, parameter), so quantities that are exactly zero are
measured on their group's scale. Errors below 1e-17, exact zeros included,
are drawn at the "exact" floor.
"""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "plots", "verification.png")

FAMILIES = ["Expected value", "Central moments", "Partial variances", "Sobol' indices",
            "Shapley effects", "Remainders"]
DISTS = ["Normal", "Uniform", "Triangular", "Kumaraswamy", "Lognormal"]
FLOOR, EPS, TOL = 1e-17, np.finfo(float).eps, 1e-10

# Reference palette (dataviz skill), light mode
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = {"value": "#2a78d6", "sensitivity": "#eb6834"}


def load():
    with open(os.path.join(HERE, "verification_results.csv")) as fh:
        rows = list(csv.DictReader(fh))
    scale = defaultdict(float)
    key = lambda r: (r["function"], r["family"], r["kind"], r["parameter"])  # noqa: E731
    for r in rows:
        r["library"], r["exact"] = float(r["library"]), float(r["exact"])
        scale[key(r)] = max(scale[key(r)], abs(r["exact"]))
    for r in rows:
        denom = max(abs(r["exact"]), 1e-12 * scale[key(r)])
        err = abs(r["library"] - r["exact"]) / denom if denom > 0 else abs(r["library"])
        r["err"] = max(err, FLOOR)
    return rows


def jitter(i, width=0.09):
    # deterministic spread within a sub-row
    return width * (((i * 0.6180339887) % 1.0) * 2 - 1)


def main():
    rows = load()
    n_val = sum(r["kind"] == "value" for r in rows)
    n_sen = len(rows) - n_val
    worst = max(r["err"] for r in rows)
    functions = sorted({r["function"] for r in rows})
    n_random = sum("random" in fn for fn in functions)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, axes = plt.subplots(1, len(DISTS), figsize=(14.5, 4.6), sharey=True)
    ypos = {fam: len(FAMILIES) - 1 - i for i, fam in enumerate(FAMILIES)}
    for ax, dist in zip(axes, DISTS):
        for kind, dy in (("value", 0.17), ("sensitivity", -0.17)):
            pts = [r for r in rows if r["distribution"] == dist and r["kind"] == kind]
            x = [r["err"] for r in pts]
            y = [ypos[r["family"]] + dy + jitter(i) for i, r in enumerate(pts)]
            ax.scatter(x, y, s=16, color=SERIES[kind], edgecolor=SURFACE, linewidth=0.5, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(FLOOR / 3, 1e-6)
        ax.set_xticks([FLOOR, 1e-15, 1e-12, 1e-9, 1e-6])
        ax.set_xticklabels(["exact", r"$10^{-15}$", r"$10^{-12}$", r"$10^{-9}$", r"$10^{-6}$"])
        ax.minorticks_off()
        for xv in (EPS, TOL):
            ax.axvline(xv, color=AXIS, linewidth=1, zorder=1)
        ax.set_yticks(range(len(FAMILIES)))
        ax.set_yticklabels(FAMILIES[::-1])
        ax.set_ylim(-0.6, len(FAMILIES) - 0.4)
        for yv in np.arange(0.5, len(FAMILIES) - 1, 1.0):
            ax.axhline(yv, color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        n_here = sum(r["distribution"] == dist for r in rows)
        f_here = len({r["function"] for r in rows if r["distribution"] == dist})
        ax.set_title(f"{dist} inputs\n{n_here} checks, {f_here} functions", fontsize=9.5, color=INK,
                     loc="left", pad=16)
        ax.set_xlabel("relative error vs exact answer", color=INK2)
    # reference-line labels, once
    axes[0].text(EPS, len(FAMILIES) - 0.45, " machine ε", color=MUTED, fontsize=8, va="bottom")
    axes[0].text(TOL, len(FAMILIES) - 0.45, " test tolerance", color=MUTED, fontsize=8, va="bottom")

    fig.suptitle("Every TSEUQLib quantity and its parameter sensitivities match the exact answer",
                 x=0.012, ha="left", fontsize=12.5, color=INK, y=0.985)
    fig.text(0.012, 0.865,
             f"{len(rows)} checks on {len(functions)} functions ({n_random} random, from a fixed "
             f"recipe; see functions.md) against an exact symbolic (SymPy) solution for the same Taylor "
             f"expansion:\n{n_val} values and {n_sen} sensitivities to distribution parameters. "
             f"Worst relative error {worst:.0e}.",
             fontsize=9, color=INK2, ha="left")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=6, color=SERIES[k],
                          markeredgecolor=SURFACE, label=lab)
               for k, lab in (("value", "value"), ("sensitivity", "sensitivity to a distribution parameter"))]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.004, 0.845), ncol=2,
               frameon=False, fontsize=8.5, labelcolor=INK2, handletextpad=0.2, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
