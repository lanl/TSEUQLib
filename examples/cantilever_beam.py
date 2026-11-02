"""A TSEUQLib workflow, start to finish, on the tip deflection of a cantilever.

    delta = 4 P L^3 / (E w h^3)      (mm, with E in GPa and lengths in m)

Five uncertain inputs, one of each distribution TSEUQLib supports:

    P  tip load (N)            normal, mean 1000, sd 100
    E  Young's modulus (GPa)   lognormal, median 200, log-sd 0.05
    L  length (m)              uniform on [1.95, 2.05]
    w  width (m)               symmetric triangular on [0.045, 0.055]
    h  height (m)              h = 0.08 + 0.04 u, u Kumaraswamy(2, 2.5) on [0, 1]

The steps an engineer follows (see the flowchart in the README):

  1. Build the Taylor series T_n of the model at the mean of the inputs: one
     evaluation of the model in OTI arithmetic gives every derivative up to
     order n. (For a finite element model they would come from a
     hypercomplex FE run, HYPAD.)
  2. Compute the mean, spread, skewness and kurtosis of T_n.
  3. Check the remainder estimate R_{n,l} = Q[T_{n+l}] - Q[T_n]: raise n
     until it is small enough for the question at hand. Look two orders
     ahead (l = 2): the next order alone can contribute little (here the
     third-order terms barely move the mean) while the one after does not.
  4. Rank the inputs: Sobol', total Sobol' and Shapley indices.
  5. Ask how the answer moves with the inputs' parameters (here: how the
     spread of the deflection responds to each input's spread). These
     converge with n too; a small net effect (the Kumaraswamy shape moves
     both the mean and the spread of h) needs a higher order.

Because delta is a product of independent factors, every quantity of the
true model is a product of one-dimensional integrals, so this script also
computes the true values (mpmath, 30 digits) and shows how T_n approaches
them. Writes examples/plots/cantilever_beam.png.

Run from the repository root: python -B examples/cantilever_beam.py
"""

import os
import sys
from itertools import combinations
from math import factorial

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pyoti.sparse as oti  # noqa: E402
from mpmath import mp  # noqa: E402

import tseuqlib as tuq  # noqa: E402

OUT = os.path.join(ROOT, "examples", "plots", "cantilever_beam.png")
NAMES = ["P", "E", "L", "w", "h"]
DISTS = np.array(["N", "LN", "U", "T", "K"])
A = [1000.0, np.log(200.0), 1.95, 0.045, 2.0]
B = [100.0, 0.05, 2.05, 0.055, 2.5]
H0, HS = 0.08, 0.04          # h = H0 + HS u
ORDERS = [1, 2, 3, 4, 5]
EL_ORDERS = [2, 3, 4]
mp.dps = 30


def deflection(x):
    """The model, as ordinary code: it runs on floats and on OTI numbers."""
    P, E, L, w, u = x
    h = H0 + HS * u
    return 4e-6 * P * L**3 / (E * w * h**3)


# ---- the TSEUQLib side ----------------------------------------------------------

def surrogate(n, a=A, b=B):
    """T_n of the deflection and the tse_uq object that integrates it."""
    mom = tuq.rv_central_moments([DISTS, np.array(a, dtype=object), np.array(b, dtype=object)], 4 * n)
    mom.compute_central_moments()
    tq = tuq.tse_uq(len(NAMES), tuq.build_rv_joint_moments(mom.rv_mu))
    return tq, tuq.tse(deflection, mom.rv_mean, n)


def moments(tq, y):
    mean = tq.expectation(y).real
    mu2, mu3, mu4 = (tq.central_moment(y, k).real for k in (2, 3, 4))
    return {"mean": mean, "sd": mu2**0.5, "skewness": mu3 / mu2**1.5, "kurtosis": mu4 / mu2**2}


def indices(tq, y):
    S = tq.sobol_indices_by_term(y)[0]
    V = tq.central_moment(y, 2).real
    return {"Sobol'": [float(np.real(S[(i + 1,)])) for i in range(len(NAMES))],
            "total Sobol'": [float(np.real(v)) for v in tq.total_sobol_indices(y)],
            "Shapley": [float(np.real(v)) / V for v in tq.shapley_values(y)[0]]}


# the spread of each input, and how to move it: (parameter label, which, input, sign)
SPREADS = [("sd of P", "b", 0), ("log-sd of E", "b", 1), ("half-width of L", "hw", 2),
           ("half-width of w", "hw", 3), ("shape b of h", "b", 4)]


def spread_elasticities(n):
    """(theta / sd) d sd / d theta for each input's spread theta, all five at
    once: each parameter gets its own OTI basis above the inputs'."""
    a = np.array(A, dtype=object)
    b = np.array(B, dtype=object)
    for j, (_, which, i) in enumerate(SPREADS):
        e = oti.e(len(NAMES) + 1 + j, order=1)
        if which == "hw":  # move both ends apart: a - e, b + e
            a[i], b[i] = a[i] - e, b[i] + e
        else:
            b[i] = b[i] + e
    mom = tuq.rv_central_moments([DISTS, a, b], 2 * n)
    mom.compute_central_moments()
    tq = tuq.tse_uq(len(NAMES), tuq.build_rv_joint_moments(mom.rv_mu))
    sd = tq.central_moment(tuq.tse(deflection, mom.rv_mean, n), 2)**0.5
    return [theta(j) / sd.real * sd.get_deriv([len(NAMES) + 1 + j]) for j in range(len(SPREADS))]


def theta(j, a=A, b=B):
    _, which, i = SPREADS[j]
    return (b[i] - a[i]) / 2 if which == "hw" else b[i]


# ---- the true model, by one-dimensional integrals ----------------------------------

def factor_moments(a, b, kmax=4):
    """E[g_i^k], k = 1..kmax, for the factors g = (P, 1/E, L^3, 1/w, 1/h^3)."""
    a, b = [mp.mpf(x) for x in a], [mp.mpf(x) for x in b]
    out = []
    # P normal: E[P^k]
    m, s = a[0], b[0]
    out.append([sum(mp.binomial(k, j) * m**(k - j) * s**j * mp.fac2(j - 1)
                    for j in range(0, k + 1, 2)) for k in range(1, kmax + 1)])
    # E lognormal: E[E^-k]
    out.append([mp.exp(-k * a[1] + k**2 * b[1]**2 / 2) for k in range(1, kmax + 1)])
    # L uniform: E[L^3k]
    out.append([(b[2]**(3 * k + 1) - a[2]**(3 * k + 1)) / ((3 * k + 1) * (b[2] - a[2])) for k in range(1, kmax + 1)])
    # w symmetric triangular: E[w^-k]
    lo, hi = a[3], b[3]
    c = (lo + hi) / 2
    pdf = lambda t: 2 * (t - lo) / ((hi - lo) * (c - lo)) if t < c else 2 * (hi - t) / ((hi - lo) * (hi - c))  # noqa: E731
    out.append([mp.quad(lambda t: t**-k * pdf(t), [lo, c, hi]) for k in range(1, kmax + 1)])
    # h = H0 + HS u, u Kumaraswamy: E[h^-3k]
    ka, kb = a[4], b[4]
    kpdf = lambda u: ka * kb * u**(ka - 1) * (1 - u**ka)**(kb - 1)  # noqa: E731
    out.append([mp.quad(lambda u: (H0 + HS * u)**(-3 * k) * kpdf(u), [0, 1]) for k in range(1, kmax + 1)])
    return out


def true_values(a=A, b=B):
    g = factor_moments(a, b)
    C = mp.mpf("4e-6")
    raw = [C**k * mp.fprod(gi[k - 1] for gi in g) for k in (1, 2, 3, 4)]
    m = raw[0]
    mu2 = raw[1] - m**2
    mu3 = raw[2] - 3 * m * raw[1] + 2 * m**3
    mu4 = raw[3] - 4 * m * raw[2] + 6 * m**2 * raw[1] - 3 * m**4
    mom = {"mean": m, "sd": mp.sqrt(mu2), "skewness": mu3 / mu2**1.5, "kurtosis": mu4 / mu2**2}
    # closed variances Var(E[delta | x_u]) of a product of independent factors
    d = len(g)
    m1sq = [gi[0]**2 for gi in g]
    m2 = [gi[1] for gi in g]
    closed = {u: C**2 * (mp.fprod(m2[i] if i in u else m1sq[i] for i in range(d)) - mp.fprod(m1sq))
              for r in range(d + 1) for u in combinations(range(d), r)}
    V = {u: sum((-1)**(len(u) - len(v)) * closed[v] for r in range(len(u) + 1) for v in combinations(u, r))
         for u in closed if u}
    idx = {"Sobol'": [float(V[(i,)] / mu2) for i in range(d)],
           "total Sobol'": [float(1 - closed[tuple(j for j in range(d) if j != i)] / mu2) for i in range(d)],
           "Shapley": [float(sum(Vu / len(u) for u, Vu in V.items() if i in u) / mu2) for i in range(d)]}
    return {k: float(v) for k, v in mom.items()}, idx, {"mean": float(m), "variance": float(mu2)}


def true_spread_elasticities():
    """Central differences of the true sd in each spread parameter."""
    out = []
    for j, (_, which, i) in enumerate(SPREADS):
        sds = []
        for sgn in (1, -1):
            a, b = list(A), list(B)
            step = 1e-6 * theta(j)
            if which == "hw":
                a[i], b[i] = a[i] - sgn * step, b[i] + sgn * step
            else:
                b[i] = b[i] + sgn * step
            sds.append(true_values(a, b)[0]["sd"])
        out.append(theta(j) / true_values()[0]["sd"] * (sds[0] - sds[1]) / (2e-6 * theta(j)))
    return out


# ---- run the workflow -------------------------------------------------------------

def main():
    true_mom, true_idx, true_mv = true_values()
    print("Step 1-2: moments of T_n (the true model's in the last row)")
    print(f"{'n':>3} " + " ".join(f"{k:>12}" for k in true_mom))
    tq5, y5 = surrogate(max(ORDERS))
    mom_n, rem_n = {}, {1: {}, 2: {}}
    for n in ORDERS:
        mom_n[n] = moments(tq5, y5.truncate_order(n + 1))
        print(f"{n:>3} " + " ".join(f"{mom_n[n][k]:12.6g}" for k in true_mom))
        for l in (1, 2):
            if n + l <= max(ORDERS):
                rem_n[l][n] = {"mean": tq5.remainder(tq5.expectation, y5, n, l).real,
                               "variance": tq5.remainder(tq5.central_moment, y5, n, l, 2).real}
    print("  true " + " ".join(f"{true_mom[k]:12.6g}" for k in true_mom))

    print("\nStep 3: remainder estimates R_{n,1}, R_{n,2} against the actual error (true - T_n)")
    for q, of_n in (("mean", lambda n: mom_n[n]["mean"]), ("variance", lambda n: mom_n[n]["sd"]**2)):
        for n in rem_n[1]:
            r2 = f"{rem_n[2][n][q]:+.3e}" if n in rem_n[2] else "       -  "
            print(f"  {q:8s} n={n}:  R_n,1={rem_n[1][n][q]:+.3e}  R_n,2={r2}  "
                  f"actual={true_mv[q] - of_n(n):+.3e}")

    n_sa = 3
    tq3, y3 = surrogate(n_sa)
    idx = indices(tq3, y3)
    print(f"\nStep 4: which inputs matter (T_{n_sa}; true model in brackets)")
    for k in idx:
        print(f"  {k:13s}" + "  ".join(f"{NAMES[i]} {idx[k][i]:.4f} [{true_idx[k][i]:.4f}]" for i in range(len(NAMES))))

    print("\nStep 5: elasticity of the deflection's sd to each input's spread, by order (true last)")
    el = {n: spread_elasticities(n) for n in EL_ORDERS}
    el_true = true_spread_elasticities()
    print(f"  {'':16s}" + "".join(f"{'T_' + str(n):>9}" for n in EL_ORDERS) + f"{'true':>9}")
    for j, (label, _, _) in enumerate(SPREADS):
        print(f"  {label:16s}" + "".join(f"{el[n][j]:+9.4f}" for n in EL_ORDERS) + f"{el_true[j]:+9.4f}")

    plot(mom_n, true_mom, rem_n, true_mv, idx, true_idx, el, el_true, n_sa)


# ---- the figure -------------------------------------------------------------------

def plot(mom_n, true_mom, rem_n, true_mv, idx, true_idx, el, el_true, n_sa):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    GRID, AXIS = "#e1e0d9", "#c3c2b7"
    SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    for ax in axes.flat:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # (a) moments converge with order
    ax = axes[0, 0]
    nudge = {"skewness": 5, "kurtosis": -5}  # their ends nearly meet
    for c, k in zip(SLOTS, true_mom):
        err = [max(abs(mom_n[n][k] - true_mom[k]) / abs(true_mom[k]), 1e-17) for n in ORDERS]
        ax.plot(ORDERS, err, color=c, linewidth=2, marker="o", markersize=5)
        ax.annotate(k, (ORDERS[-1], err[-1]), xytext=(7, nudge.get(k, 0)), textcoords="offset points",
                    color=INK2, va="center", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_xticks(ORDERS)
    ax.set_xlim(ORDERS[0] - 0.2, ORDERS[-1] + 0.9)
    ax.set_xlabel("expansion order n")
    ax.set_ylabel("relative error vs the true model")
    ax.set_title("(a) The moments of T$_n$ converge to the model's", loc="left", color=INK, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)

    # (b) is n high enough? each remainder estimate as a share of the actual error
    ax = axes[0, 1]
    tv = true_mv["variance"]
    actual = {n: tv - mom_n[n]["sd"]**2 for n in rem_n[1]}
    ax.axhline(1, color=AXIS, linewidth=1, zorder=1)
    ax.annotate("perfect estimate", (max(rem_n[1]) + 0.35, 1), xytext=(0, 4), textcoords="offset points",
                color=MUTED, fontsize=8, va="bottom", ha="right")
    for l, c, lab in ((2, SLOTS[0], "R$_{n,2}$: two orders ahead"), (1, SLOTS[1], "R$_{n,1}$: next order only")):
        ns = sorted(rem_n[l])
        share = [rem_n[l][n]["variance"] / actual[n] for n in ns]
        ax.plot(ns, share, color=c, linewidth=2, marker="o", markersize=5, zorder=3, label=lab)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK2)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(sorted(rem_n[1]))
    ax.set_xlim(0.7, max(rem_n[1]) + 0.35)
    ax.set_xlabel("expansion order n")
    ax.set_ylabel("estimated error ÷ actual error (variance)")
    ax.set_title("(b) Is n high enough? Look two orders ahead", loc="left", color=INK, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)

    # (c) which inputs matter
    ax = axes[1, 0]
    x = np.arange(len(NAMES))
    width = 0.26
    for j, (c, k) in enumerate(zip(SLOTS, idx)):
        xs = x + (j - 1) * width
        ax.bar(xs, idx[k], width=width - 0.03, color=c, label=k, zorder=2)
        ax.scatter(xs, true_idx[k], marker="_", s=120, color=INK, linewidths=1.6, zorder=3)
    ax.scatter([], [], marker="_", s=120, color=INK, linewidths=1.6, label="true model")
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES)
    ax.set_ylabel("share of the variance")
    ax.set_title(f"(c) Which inputs matter (T$_{n_sa}$)", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)

    # (d) how the spread responds to each input's spread, by order
    ax = axes[1, 1]
    labels = [sp[0] for sp in SPREADS]
    yy = np.arange(len(labels))[::-1].astype(float)
    hgt = 0.26
    for j, (c, n) in enumerate(zip(SLOTS, EL_ORDERS)):
        ax.barh(yy + (1 - j) * hgt, el[n], height=hgt - 0.03, color=c, zorder=2, label=f"T$_{n}$")
    ax.scatter(el_true, yy, marker="|", s=420, color=INK, linewidths=1.6, zorder=3)
    ax.plot([], [], marker="|", markersize=11, markeredgewidth=1.6, linestyle="", color=INK, label="true model")
    ax.set_yticks(yy)
    ax.set_yticklabels(labels)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlabel("elasticity of the deflection's sd:  (θ / sd) ∂sd/∂θ")
    ax.set_title("(d) How the spread responds to each input's spread", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor=SURFACE, framealpha=1, fontsize=8.5,
              loc="center right", labelcolor=INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)

    fig.suptitle("A TSEUQLib workflow on a cantilever's tip deflection, δ = 4PL³ / (E w h³)",
                 x=0.012, ha="left", fontsize=12.5, color=INK, y=0.985)
    fig.text(0.012, 0.935, "Five uncertain inputs (normal P, lognormal E, uniform L, triangular w, "
             "Kumaraswamy h). The true model's values come from exact one-dimensional integrals.",
             fontsize=9, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
