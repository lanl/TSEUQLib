"""Check every quantity TSEUQLib returns, and its parameter sensitivities,
against an exact symbolic answer. Writes verification_results.csv (drawn by
plot_verification.py) and functions.md (the functions and settings used).

The exact answers come from the SymPy oracle in tests/oracle.py, which
builds the same Taylor series symbolically and integrates it exactly. Each
case runs once with several distribution parameters perturbed at once (one
OTI basis each); every derivative is compared with its own single-parameter
oracle. Sensitivities are target (C): the expansion is built once at the
nominal mean, and only the input distribution moves.

Cases: two hand-picked functions per distribution (the ones used while
developing the fixes) and held-out random functions from a fixed, seeded
recipe (random_function below), not tuned by hand.

Run from the repository root:  python -B examples/verification/verify_all.py
(about 30 minutes; the Kumaraswamy and lognormal oracles are the slow part).
Each case is saved as soon as it finishes, and finished cases are skipped on a
rerun, so an interrupted run resumes; --fresh starts over, and --only ID,ID
runs just those cases (for a quick smoke test of each case type).
"""

import argparse
import csv
import os
import random
import sys
import time
from fractions import Fraction
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]

import sympy as sp  # noqa: E402

from cases import f_exp, f_three  # noqa: E402
from library_adapter import build  # noqa: E402
from oracle import TSEOracle  # noqa: E402
from tseuqlib.oti_util import pyoti_max_bases  # noqa: E402

SEED = 20261101
DISTS = {"Normal": "N", "Uniform": "U", "Triangular": "T", "Kumaraswamy": "K", "Lognormal": "LN"}
RANDOM_CASES = {"N": 4, "U": 4, "T": 4, "K": 2, "LN": 2}


# ---- the functions -------------------------------------------------------------

def _num(q, exact):
    return sp.Rational(q.numerator, q.denominator) if exact else float(q)


def make_function(terms, w, kind, shift):
    """f = sum of c * prod x_i^e_i  +  g(L),  L = sum w_i x_i, with g = exp,
    sin, or 1/(shift + L). Works on SymPy symbols and on OTI numbers."""
    def f(x, m):
        exact = m is sp
        out = 0
        for c, e in terms:
            t = _num(c, exact)
            for xi, ei in zip(x, e):
                if ei:
                    t = t * xi**ei
            out = out + t
        L = 0
        for wi, xi in zip(w, x):
            if wi:
                L = L + _num(wi, exact) * xi
        if kind == "exp":
            return out + m.exp(L)
        if kind == "sin":
            return out + m.sin(L)
        return out + 1 / (_num(shift, exact) + L)
    return f


def random_function(rng, d, n, mean):
    """The fixed recipe: 3 to 5 monomials with coefficients in [-2, 2] (steps
    of 0.1), the first of degree n + 1 or n + 2 (so the truncated terms
    matter), the others of degree 1 to n + 2, each in up to three inputs;
    plus exp, sin or a ratio of a random linear combination L (weights in
    [-1, 1]); the ratio's shift keeps it at least 2 away from 0 at the mean."""
    terms = []
    for k in range(rng.randint(3, 5)):
        deg = rng.randint(n + 1, n + 2) if k == 0 else rng.randint(1, n + 2)
        nvar = min(rng.randint(1, min(3, d)), deg)
        idx = rng.sample(range(d), nvar)
        cuts = sorted(rng.sample(range(1, deg), nvar - 1)) if nvar > 1 else []
        parts = [b - a for a, b in zip([0] + cuts, cuts + [deg])]
        e = [0] * d
        for i, p in zip(idx, parts):
            e[i] = p
        c = Fraction(rng.choice([v for v in range(-20, 21) if v]), 10)
        terms.append((c, tuple(e)))
    w = [Fraction(rng.randint(-10, 10), 10) for _ in range(d)]
    if not any(w):
        w[0] = Fraction(1, 2)
    kind = rng.choice(["exp", "sin", "ratio"])
    L0 = sum(float(wi) * mi for wi, mi in zip(w, mean))
    shift = Fraction(int(abs(L0)) + 3)
    return terms, w, kind, shift


def random_parameters(rng, code, d):
    """Nominal parameters (two decimals, so SymPy keeps them exact)."""
    r2 = lambda lo, hi: round(rng.uniform(lo, hi), 2)  # noqa: E731
    if code == "N":
        return [r2(0.5, 2.0) for _ in range(d)], [r2(0.1, 0.5) for _ in range(d)]
    if code in ("U", "T"):
        a = [r2(0.0, 1.0) for _ in range(d)]
        return a, [round(ai + r2(0.5, 1.5), 2) for ai in a]
    if code == "LN":
        return [r2(-0.5, 0.5) for _ in range(d)], [r2(0.1, 0.5) for _ in range(d)]
    return [r2(1.5, 3.5) for _ in range(d)], [r2(1.5, 3.5) for _ in range(d)]


def nominal_mean(code, a, b):
    if code == "N":
        return list(a)
    if code in ("U", "T"):
        return [(ai + bi) / 2 for ai, bi in zip(a, b)]
    if code == "LN":
        return [float(sp.exp(ai + bi**2 / 2)) for ai, bi in zip(a, b)]
    return [0.5] * len(a)  # Kumaraswamy lives on [0, 1]


def formula(f, d):
    x = sp.symbols(f"x1:{d + 1}")
    return sp.sstr(f(list(x), sp))


# ---- the checks ----------------------------------------------------------------

def _record(rows, case, family, name, lib_value, extra, params, exact_of):
    """One library result: its value and its derivative for every parameter.
    exact_of(p) returns (oracle, SymPy expression) for parameter p (None: value)."""
    for basis, p in [(None, None)] + list(zip(extra, params)):
        o, expr = exact_of(p)
        if p is None:
            got = float(lib_value.real) if hasattr(lib_value, "real") else float(lib_value)
            exact = o.value(expr)
        else:
            got = float(lib_value.get_deriv([basis])) if hasattr(lib_value, "get_deriv") else 0.0
            exact = o.deriv(expr)
        rows.append({"distribution": case["distribution"], "function": case["id"], "family": family,
                     "quantity": name, "kind": "value" if p is None else "sensitivity",
                     "parameter": "" if p is None else f"{p[0]}{p[1] + 1}",
                     "library": repr(got), "exact": repr(exact)})


def check_case(rows, case):
    """E, central moments 2-4, every partial variance and Sobol' index (read by
    label), Shapley effects, and the remainders R_{n-1,1} of E, the central
    moments (mu3, mu4 not for Kumaraswamy and lognormal: slow), partial variances and Sobol'
    indices: values and sensitivities to every perturbed parameter."""
    f, names, a, b, n, params = (case[k] for k in ("f", "names", "a", "b", "n", "params"))
    d = len(names)
    tq, y, extra = build(f, names, a, b, n, param=params)
    hi = {p: TSEOracle(lambda x: f(x, sp), names, a, b, n, param=p) for p in [None] + params}
    lo = {p: TSEOracle(lambda x: f(x, sp), names, a, b, n - 1, param=p) for p in [None] + params}
    rec = lambda family, name, val, ex: _record(rows, case, family, name, val, extra, params, ex)  # noqa: E731

    rec("Expected value", "E", tq.expectation(y), lambda p: (hi[p], hi[p].expectation()))
    for k in (2, 3, 4):
        rec("Central moments", f"mu{k}", tq.central_moment(y, k),
            lambda p, k=k: (hi[p], hi[p].central_moment(k)))
    S, V = tq.sobol_indices_by_term(y)
    for key in S:
        u = tuple(i - 1 for i in key)
        label = "".join(str(i) for i in key)
        rec("Partial variances", f"V{label}", V[key], lambda p, u=u: (hi[p], hi[p].partial_variances()[0][u]))
        rec("Sobol' indices", f"S{label}", S[key], lambda p, u=u: (hi[p], hi[p].sobol_indices()[u]))
    shap = tq.shapley_values_v2(y)[0]
    for i in range(d):
        rec("Shapley effects", f"phi{i + 1}", shap[i],
            lambda p, i=i: (hi[p], hi[p].shapley_effects()[i] * hi[p].central_moment(2)))

    # remainders R_{n-1,1} = Q[T_n] - Q[T_{n-1}]
    rec("Remainders", "R E", tq.expectation_remainder(y, n - 1, 1),
        lambda p: (hi[p], hi[p].expectation() - lo[p].expectation()))
    for k in ((2,) if names[0] in ("K", "LN") else (2, 3, 4)):
        rec("Remainders", f"R mu{k}", tq.central_moment_remainder(y, n - 1, 1, k),
            lambda p, k=k: (hi[p], hi[p].central_moment(k) - lo[p].central_moment(k)))
    S_rem, V_rem = tq.sobol_indices_remainder(y, n - 1, 1)
    for r in range(1, len(V_rem)):
        for key, s, v in zip(tq.term_labels(r), S_rem[r], V_rem[r]):
            u = tuple(i - 1 for i in key)
            label = "".join(str(i) for i in key)
            rec("Remainders", f"R V{label}", v, lambda p, u=u: (
                hi[p], hi[p].partial_variances()[0][u] - lo[p].partial_variances()[0][u]))
            rec("Remainders", f"R S{label}", s, lambda p, u=u: (
                hi[p], hi[p].sobol_indices()[u] - lo[p].sobol_indices()[u]))


# ---- the cases -----------------------------------------------------------------

def hand_picked():
    """The two functions used while developing the fixes, per distribution."""
    nominal = {"N": ([1.0, 2.0, 0.5], [0.5, 0.3, 0.4]), "U": ([0.5, 1.5, 0.2], [1.5, 2.5, 1.0]),
               "T": ([0.5, 1.5, 0.2], [1.5, 2.5, 1.0]), "K": ([2.0, 3.0, 2.5], [2.5, 2.0, 3.0]),
               "LN": ([0.1, -0.3, 0.2], [0.2, 0.3, 0.25])}
    out = []
    for dist, code in DISTS.items():
        a, b = nominal[code]
        out.append(dict(id=f"{code}-hand-1", origin="hand-picked", distribution=dist, f=f_exp,
                        names=[code] * 2, a=a[:2], b=b[:2], n=2, params=[("a", 0), ("b", 0), ("b", 1)],
                        formula="exp(x1/2)*x2 + x2**3"))
        # n = 3 brings in the kept-input x d mu/dp cross term (z1 z2^2)
        n = 2 if code in ("K", "LN") else 3
        out.append(dict(id=f"{code}-hand-2", origin="hand-picked", distribution=dist, f=f_three,
                        names=[code] * 3, a=a, b=b, n=n, params=[("a", 0), ("b", 1)],
                        formula="x1**3 + x1*x2**2 + x2*x3 + exp(x3/3)"))
    return out


def held_out():
    """Random functions and settings from the seeded recipe."""
    rng = random.Random(SEED)
    out = []
    for dist, code in DISTS.items():
        for k in range(RANDOM_CASES[code]):
            slow = code in ("K", "LN")  # SymPy oracles with Beta / exp of the parameters
            d = 2 if slow else rng.randint(2, 4)
            n = 2 if (slow or d == 4) else rng.randint(2, 3)
            a, b = random_parameters(rng, code, d)
            terms, w, kind, shift = random_function(rng, d, n, nominal_mean(code, a, b))
            f = make_function(terms, w, kind, shift)
            # as many parameters as pyoti's basis limit allows at order 4n (mu4), up
            # to 3 (2 for the slow oracles, which grow quickly with each one)
            n_par = max(1, min(2 if slow else 3, pyoti_max_bases(4 * n) - d))
            params = rng.sample([(wh, i) for wh in ("a", "b") for i in range(d)], n_par)
            out.append(dict(id=f"{code}-random-{k + 1}", origin="random", distribution=dist, f=f,
                            names=[code] * d, a=a, b=b, n=n, params=params, formula=formula(f, d)))
    return out


def write_functions_md(cases, path):
    lines = ["# Functions in the verification figure", "",
             "Written by `verify_all.py`; do not edit by hand. Inputs are independent, all with the "
             "distribution named. Parameters: N (mean a, sd b), U and T (bounds [a, b]), K (shapes "
             "a, b on [0, 1]), LN (mean a and sd b of log x). The perturbed parameters, all at once, carry the sensitivities.", "",
             "| id | origin | inputs | n | f(x) | a | b | perturbed |",
             "|---|---|---|---|---|---|---|---|"]
    for c in cases:
        pert = ", ".join(f"{wh}{i + 1}" for wh, i in c["params"])
        lines.append(f"| {c['id']} | {c['origin']} | {len(c['names'])} {c['names'][0]} | {c['n']} | "
                     f"`{c['formula']}` | {c['a']} | {c['b']} | {pert} |")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


FIELDS = ["distribution", "function", "family", "quantity", "kind", "parameter", "library", "exact"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fresh", action="store_true", help="discard saved results and start over")
    ap.add_argument("--only", default="", help="comma-separated case ids to run")
    args = ap.parse_args()
    cases = hand_picked() + held_out()
    write_functions_md(cases, os.path.join(HERE, "functions.md"))
    out = os.path.join(HERE, "verification_results.csv")
    if args.fresh and os.path.exists(out):
        os.remove(out)
    done = set()
    if os.path.exists(out):
        with open(out) as fh:
            done = {r["function"] for r in csv.DictReader(fh)}
    only = set(args.only.split(",")) - {""}
    t0 = time.time()
    for c in cases:
        if (only and c["id"] not in only) or c["id"] in done:
            continue
        t1 = time.time()
        rows = []
        check_case(rows, c)
        new = not os.path.exists(out)
        with open(out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"{c['id']:14s} d={len(c['names'])} n={c['n']} {len(rows):4d} checks "
              f"({time.time() - t1:5.0f} s)", flush=True)
    with open(out) as fh:
        n_rows = sum(1 for _ in csv.DictReader(fh))
    print(f"{out}: {n_rows} checks; this run took {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
