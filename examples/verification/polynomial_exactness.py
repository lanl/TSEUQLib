"""Exactness on polynomials: for a polynomial model of degree p, a Taylor
series of order n >= p is the model itself, so every quantity TSEUQLib
returns should equal the model's own exact value; below p it shows what
truncating costs. Writes polynomial_results.csv (plot it with
plot_polynomial_exactness.py).

Each polynomial is f(x) = sum_alpha c_alpha prod ((x_i - m_i) / s_i)^alpha_i,
with a random coefficient on every monomial up to degree p (m_i, s_i: the
mean and standard deviation of input i). The library sees f only as a
function of x. The reference expands the same polynomial algebraically and
integrates it against each input's exact central moments (SymPy, evaluated
to 50 digits): the mean and central moments 2-4 directly, Sobol' indices
from conditional variances by inclusion-exclusion, Shapley effects from
the permutation definition. It builds no Taylor series and shares no
formula with the library.

Run from the repository root: python -B examples/verification/polynomial_exactness.py
"""

import argparse
import csv
import os
import random
import sys
import time
from itertools import combinations, combinations_with_replacement
from math import factorial

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]

import sympy as sp  # noqa: E402
from mpmath import mp, mpf  # noqa: E402

from library_adapter import build, real  # noqa: E402
from oracle import _rv_mean_and_central_moments  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "polynomial_results.csv")
SEED = 20261102
DISTS = {"Normal": "N", "Uniform": "U", "Triangular": "T", "Kumaraswamy": "K", "Lognormal": "LN"}
DEGREES = [2, 3, 4, 5]
DIMS = [2, 3]
mp.dps = 50


# ---- the polynomials --------------------------------------------------------------

def random_parameters(rng, code, d):
    """Nominal parameters, two decimals (the recipe of verify_all.py)."""
    r2 = lambda lo, hi: round(rng.uniform(lo, hi), 2)  # noqa: E731
    if code == "N":
        return [r2(0.5, 2.0) for _ in range(d)], [r2(0.1, 0.5) for _ in range(d)]
    if code in ("U", "T"):
        a = [r2(0.0, 1.0) for _ in range(d)]
        return a, [round(ai + r2(0.5, 1.5), 2) for ai in a]
    if code == "LN":
        return [r2(-0.5, 0.5) for _ in range(d)], [r2(0.1, 0.5) for _ in range(d)]
    return [r2(1.5, 3.5) for _ in range(d)], [r2(1.5, 3.5) for _ in range(d)]


def monomials(d, p):
    out = [(0,) * d]
    for k in range(1, p + 1):
        for dirs in combinations_with_replacement(range(d), k):
            out.append(tuple(dirs.count(i) for i in range(d)))
    return out


def make_polynomial(rng, d, p, m, s):
    """{alpha: c_alpha} and f(x, alg) evaluating it (alg unused: + - * only)."""
    coeffs = {alpha: round(rng.gauss(0.0, 1.0), 3) for alpha in monomials(d, p)}

    def f(x, alg=None):
        u = [(x[i] - m[i]) / s[i] for i in range(d)]
        powers = [[1.0] for _ in range(d)]
        for i in range(d):
            for _ in range(p):
                powers[i].append(powers[i][-1] * u[i])
        total = 0.0
        for alpha, c in coeffs.items():
            term = c
            for i, k in enumerate(alpha):
                if k:
                    term = term * powers[i][k]
            total = total + term
        return total
    return coeffs, f


# ---- the reference: polynomials in z = x - E[x] as {exponents: mpf} -----------------

def pmul(p, q):
    out = {}
    for a, c in p.items():
        for b, e in q.items():
            k = tuple(x + y for x, y in zip(a, b))
            out[k] = out.get(k, 0) + c * e
    return out


def pexpect(p, mom, keep=()):
    """Integrate out every input not in keep: z_i^k -> E[z_i^k]."""
    out = {}
    for a, c in p.items():
        w = c
        for i, k in enumerate(a):
            if i not in keep and k:
                w = w * mom[i][k]
        key = tuple(k if i in keep else 0 for i, k in enumerate(a))
        out[key] = out.get(key, 0) + w
    return out


def pmean(p, mom):
    return pexpect(p, mom).get((0,) * len(mom), mpf(0))


def pcentral(p, mom, k):
    d = len(mom)
    c = dict(p)
    c[(0,) * d] = c.get((0,) * d, mpf(0)) - pmean(p, mom)
    out = {(0,) * d: mpf(1)}
    for _ in range(k):
        out = pmul(out, c)
    return pmean(out, mom)


def reference(coeffs, d, p, m, s, exact_mean, mom):
    """Exact quantities of f; the library's float m, s are kept exactly, so
    x_i - m_i = z_i + (E[x_i] - m_i)."""
    u = [{tuple(int(j == i) for j in range(d)): 1 / mpf(s[i]),
          (0,) * d: (exact_mean[i] - mpf(m[i])) / mpf(s[i])} for i in range(d)]
    upow = []
    for i in range(d):
        pw = [{(0,) * d: mpf(1)}]
        for _ in range(p):
            pw.append(pmul(pw[-1], u[i]))
        upow.append(pw)
    f = {}
    for alpha, c in coeffs.items():
        term = {(0,) * d: mpf(c)}
        for i, k in enumerate(alpha):
            if k:
                term = pmul(term, upow[i][k])
        for key, v in term.items():
            f[key] = f.get(key, 0) + v
    out = {"E": pmean(f, mom)}
    for k in (2, 3, 4):
        out[f"mu{k}"] = pcentral(f, mom, k)
    V = out["mu2"]
    closed = {v: pcentral(pexpect(f, mom, keep=v), mom, 2)
              for r in range(d + 1) for v in combinations(range(d), r)}
    for r in range(1, d + 1):
        for u_ in combinations(range(d), r):
            Vu = sum((-1) ** (r - len(v)) * closed[v] for k in range(r + 1) for v in combinations(u_, k))
            out[("S",) + tuple(i + 1 for i in u_)] = Vu / V
    for i in range(d):
        others = [j for j in range(d) if j != i]
        phi = 0
        for r in range(d):
            w = mpf(factorial(r) * factorial(d - r - 1)) / factorial(d)
            for v in combinations(others, r):
                phi += w * (closed[tuple(sorted(v + (i,)))] - closed[v])
        out[("Sh", i + 1)] = phi / V
    return out


def input_moments(code, a, b, kmax):
    """Exact mean and central moments E[z^k], k = 0..kmax, at 50 digits."""
    A = [sp.Rational(str(x)) for x in a]
    B = [sp.Rational(str(x)) for x in b]
    means, mom = [], []
    for ai, bi in zip(A, B):
        mean, mu = _rv_mean_and_central_moments(code, ai, bi, kmax)
        means.append(mpf(str(sp.N(mean, 60))))
        mom.append([mpf(1)] + [mpf(str(sp.N(mk, 60))) for mk in mu])
    return means, mom


# ---- the library ----------------------------------------------------------------

def library(f, code, d, a, b, n):
    tq, y, _ = build(f, [code] * d, a, b, n)
    out = {"E": real(tq.expectation(y))}
    for k in (2, 3, 4):
        out[f"mu{k}"] = real(tq.central_moment(y, k))
    S, _ = tq.sobol_indices_by_term(y)
    for key, v in S.items():
        out[("S",) + tuple(key)] = real(v)
    shap = tq.shapley_values_v2(y)
    for i in range(d):
        out[("Sh", i + 1)] = real(shap[0, i]) / out["mu2"]
    return out


FAMILY = {"E": "Expected value", "mu2": "Variance", "mu3": "Third central moment",
          "mu4": "Fourth central moment", "S": "Sobol' indices", "Sh": "Shapley effects"}


def name_of(key):
    if isinstance(key, str):
        return key
    tag = "S" if key[0] == "S" else "Sh"
    return tag + "".join(str(i) for i in key[1:])


def cases(quick=False):
    rng = random.Random(SEED)
    out = []
    for dist, code in DISTS.items():
        for p in DEGREES:
            for d in DIMS:
                a, b = random_parameters(rng, code, d)
                seed = rng.randrange(2**31)
                if quick and (p > 3 or d > 2):
                    continue
                out.append((dist, code, p, d, a, b, seed))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="smoke test: degrees 2-3, two inputs")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    t_all = time.time()
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["distribution", "inputs", "degree", "order", "family", "quantity", "library", "exact"])
        for dist, code, p, d, a, b, seed in cases(args.quick):
            t0 = time.time()
            exact_mean, mom = input_moments(code, a, b, 4 * p)
            m = [float(x) for x in exact_mean]
            s = [float(mp.sqrt(mom[i][2])) for i in range(d)]
            coeffs, f = make_polynomial(random.Random(seed), d, p, m, s)
            ref = reference(coeffs, d, p, m, s, exact_mean, mom)
            for n in range(1, p + 2):
                lib = library(f, code, d, a, b, n)
                for key, ex in ref.items():
                    fam = FAMILY[key if isinstance(key, str) else key[0]]
                    w.writerow([dist, d, p, n, fam, name_of(key), repr(lib[key]), mp.nstr(ex, 30)])
            fh.flush()
            print(f"{dist:12s} p={p} d={d}  {time.time() - t0:6.1f} s", flush=True)
    print(f"wrote {args.out} in {time.time() - t_all:.0f} s")


if __name__ == "__main__":
    main()
