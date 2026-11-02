"""tse_moments (the closed-form path) against a direct reference: the Taylor
polynomial expanded term by term and integrated with the inputs' central
moments, independent of both library paths and of pyoti's basis limits.

Every monomial gets a random coefficient, scaled by the input standard
deviations so each order contributes about equally (a wrong top-order term
shows as an O(1) error). Skewed inputs (K, LN) keep the odd-moment terms
alive. Every quantity is checked for several expansion points at once, the
way the closed forms take them. Products of four cubics reach degree 12, so
mu4 of T3 can couple up to six inputs; the six-input case covers that."""

import contextlib
import io
from itertools import combinations, combinations_with_replacement

import numpy as np
import pytest

from library_adapter import oti, rv
from tseuqlib.tse_moments import tse_central_moment

REL = 1e-10
N_POINTS = 3

# (id, names, a, b, highest order the case is checked at)
CASES = [
    ("skewed-4", ["K", "LN", "K", "LN"], [2.0, 0.0, 3.0, -0.3], [2.5, 0.3, 1.8, 0.2]),
    ("symmetric-4", ["N", "U", "T", "N"], [1.0, 0.5, 0.2, -1.0], [0.5, 1.5, 1.0, 0.3]),
    ("skewed-6", ["K", "LN", "K", "LN", "K", "LN"], [2.0, 0.0, 3.0, -0.3, 1.5, 0.2],
     [2.5, 0.3, 1.8, 0.2, 4.0, 0.25]),
]


# ---- the reference: polynomials as {exponent tuple: coefficient} ---------------

def _mul(p, q):
    out = {}
    for a, c in p.items():
        for b, e in q.items():
            k = tuple(x + y for x, y in zip(a, b))
            out[k] = out.get(k, 0.0) + c * e
    return out


def _expect(p, mu, keep=()):
    """Integrate out every input not in keep: z_i^k -> E[z_i^k]."""
    out = {}
    for a, c in p.items():
        w = c
        for i, k in enumerate(a):
            if i not in keep and k:
                w *= mu[k - 1][i]
        key = tuple(k if i in keep else 0 for i, k in enumerate(a))
        out[key] = out.get(key, 0.0) + w
    return out


def _mean(p, mu):
    return _expect(p, mu).get((0,) * len(mu[0]), 0.0)


def _central(p, mu, k):
    d = len(mu[0])
    c = dict(p)
    c[(0,) * d] = c.get((0,) * d, 0.0) - _mean(p, mu)
    out = {(0,) * d: 1.0}
    for _ in range(k):
        out = _mul(out, c)
    return _mean(out, mu)


def _partial_variance(p, mu, u):
    """V_u = sum over v in u of (-1)^(|u|-|v|) Var(E[T | x_v])."""
    return sum((-1) ** (len(u) - r) * _central(_expect(p, mu, keep=v), mu, 2)
               for r in range(len(u) + 1) for v in combinations(u, r))


def _truncate(p, k):
    return {a: c for a, c in p.items() if sum(a) <= k}


# ---- the cases ----------------------------------------------------------------

_BUILT = {}


def _build(case):
    """Central moments 1-12 of the inputs, N_POINTS random fifth-order
    polynomials, and the closed-form object holding them all."""
    if case[0] in _BUILT:
        return _BUILT[case[0]]
    _, names, a, b = case
    d = len(names)
    with contextlib.redirect_stdout(io.StringIO()):
        mom = rv.rv_central_moments([np.array(names), np.array(a, dtype=object),
                                     np.array(b, dtype=object)], 12)
        mom.compute_central_moments()
    mu = [np.array(m, dtype=float) for m in mom.rv_mu]
    sd = np.sqrt(mu[1])
    rng = np.random.default_rng(20261102)
    e = [oti.e(i + 1, order=5) for i in range(d)]
    polys, ys = [], []
    for _ in range(N_POINTS):
        p = {(0,) * d: rng.normal()}
        y = p[(0,) * d] + 0 * e[0]
        for k in range(1, 6):
            for dirs in combinations_with_replacement(range(d), k):
                alpha = tuple(dirs.count(i) for i in range(d))
                c = rng.normal() / np.prod(sd ** np.array(alpha))
                p[alpha] = c
                t = c + 0 * e[0]
                for i in dirs:
                    t = t * e[i]
                y = y + t
        polys.append(p)
        ys.append(y)
    tc = tse_central_moment(ys, d, 5, mu)
    _BUILT[case[0]] = (tc, polys, mu, d)
    return _BUILT[case[0]]


def _close(got, want, scale, msg):
    assert got == pytest.approx(want, rel=REL, abs=REL * abs(scale)), msg


EV = {1: "tseEvFirst", 2: "tseEvSecond", 3: "tseEvThird", 4: "tseEvFourth", 5: "tseEvFifth"}
TCM = {1: "tseTcmFirst", 2: "tseTcmSecond", 3: "tseTcmThird"}
FCM = {1: "tseFcmFirst", 2: "tseFcmSecond", 3: "tseFcmThird"}
VAR = {1: "tseVarFirst", 2: "tseVarSecond", 3: "tseVarThird", 4: "tseVarFourth"}

# the six-input case is there for mu4 of T3 (and mu3, Var with it); the
# lower orders are covered by the four-input cases
SIX_ONLY = {"tseTcmThird", "tseFcmThird", "tseVarThird"}


def _params(table):
    out = []
    for case in CASES:
        for k, meth in table.items():
            if case[0] == "skewed-6" and meth not in SIX_ONLY:
                continue
            out.append(pytest.param(case, k, meth, id=f"{case[0]}-{meth}"))
    return out


@pytest.mark.parametrize("case,k,meth", _params(EV))
def test_expected_value(case, k, meth):
    tc, polys, mu, d = _build(case)
    got = getattr(tc, meth)()
    for p, poly in enumerate(polys):
        T = _truncate(poly, k)
        _close(got[p], _mean(T, mu), _central(T, mu, 2) ** 0.5, f"point {p}")


@pytest.mark.parametrize("case,k,meth", _params({**TCM, **{k + 10: m for k, m in FCM.items()}}))
def test_third_and_fourth_central_moments(case, k, meth):
    tc, polys, mu, d = _build(case)
    order, kind = (k, 3) if k < 10 else (k - 10, 4)
    got = getattr(tc, meth)()
    for p, poly in enumerate(polys):
        T = _truncate(poly, order)
        var = _central(T, mu, 2)
        _close(got[p], _central(T, mu, kind), var ** (kind / 2), f"point {p}")


@pytest.mark.parametrize("case,k,meth", _params(VAR))
def test_variance_and_partial_variances(case, k, meth):
    """The variance, and for orders 1-3 every partial variance and Sobol'
    index the method returns (single inputs, pairs i<j, triples i<j<k)."""
    tc, polys, mu, d = _build(case)
    out = getattr(tc, meth)()
    var = out if k == 4 else out[0][0]
    for p, poly in enumerate(polys):
        T = _truncate(poly, k)
        V = _central(T, mu, 2)
        _close(var[p], V, V, f"variance, point {p}")
        if k == 4:
            continue  # tseVarFourth returns the variance only
        S_arrays, V_arrays = out[1], out[2]
        for r in range(1, k + 1):
            for u in combinations(range(d), r):
                Vu = _partial_variance(T, mu, u)
                idx = (p,) + u
                _close(V_arrays[r - 1][idx], Vu, V, f"V{u}, point {p}")
                _close(S_arrays[r - 1][idx], Vu / V, 1.0, f"S{u}, point {p}")


def test_reference_is_exact_on_a_known_case():
    """The reference itself, by hand: T = z1 (1 + z2) with z1, z2 standard
    normal has E = 0, Var = E[z1^2] E[(1 + z2)^2] = 2, mu3 = E[z1^3] (...)
    = 0, mu4 = E[z1^4] E[(1 + z2)^4] = 3 * 10 = 30, and V1 = Var(z1) = 1,
    V2 = 0, V12 = 2 - 1 - 0 = 1."""
    mu = [np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([0.0, 0.0]),
          np.array([3.0, 3.0]), np.array([0.0, 0.0]), np.array([15.0, 15.0])]
    T = {(1, 0): 1.0, (1, 1): 1.0}
    assert _mean(T, mu) == 0.0
    assert _central(T, mu, 2) == 2.0
    assert _central(T, mu, 3) == 0.0
    assert _central(T, mu, 4) == 30.0
    assert _partial_variance(T, mu, (0,)) == 1.0
    assert _partial_variance(T, mu, (1,)) == 0.0
    assert _partial_variance(T, mu, (0, 1)) == 1.0
