"""Are TSEUQLib's distribution-parameter sensitivities correct?

TSEUQLib differentiates the moments of the *current* surrogate T_n, built once
at the nominal input means, with respect to the parameters of the input
distributions (no new model evaluations, no higher derivatives of f).

This example checks that independently of TSEUQLib's moment code, by Monte
Carlo with common random numbers:
  1. read the Taylor coefficients of the surrogate (the OTI number y);
  2. draw standard samples once (z ~ N(0,1), u ~ U(0,1));
  3. map them to the inputs at p0 + h and p0 - h, using the same draws;
  4. evaluate the fixed surrogate on both sample sets and difference the
     sample moments.
Batch means give a standard error, so the OTI value should sit within a few
standard errors of the Monte Carlo estimate.

Run from the repository root:
    python -B examples/sensitivity_check.py
"""

import contextlib
import io
import os
import sys
from itertools import combinations_with_replacement
from math import factorial

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pyoti.sparse as oti  # noqa: E402

import tseuqlib.oti_moments as moti  # noqa: E402
import tseuqlib.oti_util as uoti  # noqa: E402
import tseuqlib.rv_moments as rv  # noqa: E402


def model(x, m=oti):
    return x[0]**3 + x[0] * x[1]**2 + x[1] * x[2] + m.exp(x[2] / 3)


# Inputs: N(mean a, sd b), N(mean a, sd b), U(a, b)
NAMES = ["N", "N", "U"]
A = [1.0, 2.0, 0.5]
B = [0.5, 0.3, 1.5]
TSE_ORDER = 2
PARAMS = [("a", 0, "mean of x1"), ("b", 0, "sd of x1"), ("b", 2, "upper bound of x3")]


def library_sensitivities(which, i):
    """E and Var of the surrogate, and their derivatives from TSEUQLib (OTI)."""
    d = len(NAMES)
    a = np.array(A, dtype=object)
    b = np.array(B, dtype=object)
    (a if which == "a" else b)[i] += oti.e(d + 1, order=1)
    with contextlib.redirect_stdout(io.StringIO()):
        mom = rv.rv_central_moments([np.array(NAMES), a, b], 4 * TSE_ORDER)
        mom.compute_central_moments()
    x = [mom.rv_mean[j] + oti.e(j + 1, order=TSE_ORDER) for j in range(d)]
    y = model(x)
    tq = moti.tse_uq(d, uoti.build_rv_joint_moments(mom.rv_mu), extra_bases=[d + 1])
    E = tq.expectation(y)
    V = tq.central_moment(y, 2)
    return E.get_deriv([d + 1]), V.get_deriv([d + 1])


def surrogate_coefficients():
    """Taylor coefficients c_alpha of the surrogate at the nominal means."""
    d = len(NAMES)
    means = [A[0], A[1], (A[2] + B[2]) / 2]
    x = [means[j] + oti.e(j + 1, order=TSE_ORDER) for j in range(d)]
    y = model(x)
    coeffs = {(0,) * d: y.real}
    for k in range(1, TSE_ORDER + 1):
        for dirs in combinations_with_replacement(range(1, d + 1), k):
            alpha = tuple(dirs.count(j) for j in range(1, d + 1))
            coeffs[alpha] = y.get_deriv(list(dirs)) / np.prod([factorial(ai) for ai in alpha])
    return coeffs, np.array(means)


def evaluate_surrogate(coeffs, m0, X):
    """The fixed polynomial sum c_alpha (x - m0)^alpha on samples X (N x d)."""
    Z = X - m0
    out = np.zeros(len(X))
    for alpha, c in coeffs.items():
        term = np.full(len(X), c)
        for j, aj in enumerate(alpha):
            if aj:
                term *= Z[:, j]**aj
        out += term
    return out


def inputs(a, b, z, u):
    return np.column_stack([a[0] + b[0] * z[:, 0], a[1] + b[1] * z[:, 1], a[2] + (b[2] - a[2]) * u])


def monte_carlo_sensitivities(which, i, coeffs, m0, n_samples=2_000_000, n_batches=40, h=1e-6, seed=0):
    """Common-random-number FD of the sample mean and variance, with batch errors."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_samples, 2))
    u = rng.random(n_samples)
    ys = []
    for s in (+1, -1):
        a, b = list(A), list(B)
        (a if which == "a" else b)[i] += s * h
        ys.append(evaluate_surrogate(coeffs, m0, inputs(a, b, z, u)))
    dE, dV = [], []
    for yp, ym in zip(np.array_split(ys[0], n_batches), np.array_split(ys[1], n_batches)):
        dE.append((yp.mean() - ym.mean()) / (2 * h))
        dV.append((yp.var() - ym.var()) / (2 * h))
    se = lambda v: np.std(v, ddof=1) / np.sqrt(n_batches)
    return np.mean(dE), se(dE), np.mean(dV), se(dV)


def main():
    coeffs, m0 = surrogate_coefficients()
    print("Sensitivities of the second-order surrogate's moments to input-distribution parameters")
    print("TSEUQLib (OTI) against Monte Carlo with common random numbers (2e6 samples, 40 batches)\n")
    print(f"{'parameter':20s} {'moment':6s} {'TSEUQLib':>12s} {'Monte Carlo':>22s} {'difference / SE':>16s}")
    for which, i, label in PARAMS:
        dE, dV = library_sensitivities(which, i)
        mE, sE, mV, sV = monte_carlo_sensitivities(which, i, coeffs, m0)
        for name, lib, mc, se in (("E", dE, mE, sE), ("Var", dV, mV, sV)):
            print(f"{label:20s} {name:6s} {lib:12.6f} {mc:12.6f} +/- {se:7.6f} {(lib - mc) / se:16.2f}")


if __name__ == "__main__":
    main()
