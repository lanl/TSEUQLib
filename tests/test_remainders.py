"""Remainders R_{n,l} = Q[T_{n+l}] - Q[T_n] against the exact oracle: the
polynomial itself (tse_remainder), and the expectation, central moments,
partial variances and Sobol' indices of it, as values and as parameter
sensitivities (target C). Two parameters at once, each on its own basis."""

from itertools import combinations

import pytest
import sympy as sp

from cases import f_cubic, f_exp, f_three
from library_adapter import _taylor_coefficients, build, oti_deriv, real
from oracle import TSEOracle

REL = 1e-10

_ORACLES = {}


def _oracle(case, n, param=None):
    key = (case[0], n, param)
    if key not in _ORACLES:
        _, f, names, a, b = case[:5]
        _ORACLES[key] = TSEOracle(lambda x: f(x, sp), names, a, b, n, param=param)
    return _ORACLES[key]


def _check(g, hi, lo, expr_hi, expr_lo, basis, msg):
    """Library entry g against oracle value (basis None) or derivative."""
    if basis is None:
        e = hi.value(expr_hi) - lo.value(expr_lo)
        v = real(g) if hasattr(g, "real") else float(g)
    else:
        e = hi.deriv(expr_hi) - lo.deriv(expr_lo)
        v = oti_deriv(g, [basis]) if hasattr(g, "get_deriv") else 0.0
    assert v == pytest.approx(e, rel=REL, abs=1e-12), msg


# ---- the remainder polynomial ------------------------------------------------

TSE_CASE = ("N-exp", f_exp, ["N", "N"], [1.0, 2.0], [0.5, 0.3])


@pytest.mark.parametrize("n,l", [(0, 1), (1, 1), (2, 1), (1, 2)], ids=lambda v: str(v))
def test_tse_remainder(n, l):
    """tse_remainder(y, n, l) is T_{n+l} - T_n, coefficient by coefficient
    (it returned T_{n+l-1} - T_{n-1} until 2026-11-01)."""
    _, f, names, a, b = TSE_CASE
    d = len(names)
    tq, y, _ = build(f, names, a, b, n + l)
    got = _taylor_coefficients(tq.tse_remainder(y, n, l), d, n + l)
    hi, lo = _oracle(TSE_CASE, n + l), _oracle(TSE_CASE, n)
    want = sp.Poly(sp.expand(hi.T - lo.T), *hi.z).as_dict()
    for alpha, c in got.items():
        assert c == pytest.approx(float(want.get(alpha, 0)), rel=REL, abs=1e-12), f"coefficient {alpha}"


# ---- remainders of the moments -------------------------------------------------

REM_CASES = [
    ("N-exp-n1-l1", f_exp, ["N", "N"], [1.0, 2.0], [0.5, 0.3], 1, 1),
    ("U-cubic-n2-l1", f_cubic, ["U", "U"], [0.5, 1.5], [1.5, 2.5], 2, 1),
]
REM_PARAMS = [("a", 0), ("b", 1)]

MOMENT_REMAINDERS = {
    "E": (lambda tq, y, n, l: tq.expectation_remainder(y, n, l), lambda o: o.expectation()),
    "mu2": (lambda tq, y, n, l: tq.central_moment_remainder(y, n, l, 2), lambda o: o.central_moment(2)),
    "mu3": (lambda tq, y, n, l: tq.central_moment_remainder(y, n, l, 3), lambda o: o.central_moment(3)),
    "mu4": (lambda tq, y, n, l: tq.central_moment_remainder(y, n, l, 4), lambda o: o.central_moment(4)),
}


@pytest.mark.parametrize("q", MOMENT_REMAINDERS)
@pytest.mark.parametrize("case", REM_CASES, ids=[c[0] for c in REM_CASES])
def test_moment_remainder(case, q):
    _, f, names, a, b, n, l = case
    qlib, qorc = MOMENT_REMAINDERS[q]
    tq, y, extra = build(f, names, a, b, n + l, param=REM_PARAMS)
    g = qlib(tq, y, n, l)
    for basis, p in [(None, None)] + list(zip(extra, REM_PARAMS)):
        hi, lo = _oracle(case, n + l, p), _oracle(case, n, p)
        _check(g, hi, lo, qorc(hi), qorc(lo), basis, f"{q} {p or 'value'}")


# ---- remainders of partial variances and Sobol' indices --------------------------

# n = 1 pads the lower expansion's orders above 1; n = 2, l = 1 brings in the
# e_u * e_p cross term (z1 z2^2 with the spread of x2 as the parameter).
SOBOL_REM_CASES = [
    ("NUT-three-n1-l1", f_three, ["N", "U", "T"], [1.0, 1.5, 0.2], [0.5, 2.5, 1.0], 1, 1),
    ("NUT-three-n2-l1", f_three, ["N", "U", "T"], [1.0, 1.5, 0.2], [0.5, 2.5, 1.0], 2, 1),
]
SOBOL_REM_PARAMS = [("a", 1), ("b", 2)]


@pytest.mark.parametrize("case", SOBOL_REM_CASES, ids=[c[0] for c in SOBOL_REM_CASES])
def test_sobol_remainder(case):
    _, f, names, a, b, n, l = case
    d = len(names)
    tq, y, extra = build(f, names, a, b, n + l, param=SOBOL_REM_PARAMS)
    S_rem, V_rem = tq.sobol_indices_remainder(y, n, l)
    for basis, p in [(None, None)] + list(zip(extra, SOBOL_REM_PARAMS)):
        hi, lo = _oracle(case, n + l, p), _oracle(case, n, p)
        S_hi, S_lo = hi.sobol_indices(), lo.sobol_indices()
        (V_hi, _), (V_lo, _) = hi.partial_variances(), lo.partial_variances()
        for r in range(1, d + 1):
            for j, u in enumerate(combinations(range(d), r)):
                _check(S_rem[r][j], hi, lo, S_hi[u], S_lo[u], basis, f"S{u} {p or 'value'}")
                _check(V_rem[r][j], hi, lo, V_hi[u], V_lo[u], basis, f"V{u} {p or 'value'}")


@pytest.mark.parametrize("n,l", [(0, 1), (1, 1), (2, 1), (1, 2)], ids=lambda v: str(v))
def test_oti_util_tse_remainder(n, l):
    """The package-level oti_util.tse_remainder (exported as
    tseuqlib.tse_remainder) agrees with the method; it had the same
    off-by-one until 2026-11-01."""
    from library_adapter import uoti
    _, f, names, a, b = TSE_CASE
    d = len(names)
    tq, y, _ = build(f, names, a, b, n + l)
    got = _taylor_coefficients(uoti.tse_remainder(y, n, l), d, n + l)
    want = _taylor_coefficients(tq.tse_remainder(y, n, l), d, n + l)
    assert got == pytest.approx(want, rel=1e-14, abs=1e-14)


def test_oti_util_tse_remainder_past_order():
    _, f, names, a, b = TSE_CASE
    from library_adapter import uoti
    _, y, _ = build(f, names, a, b, 2)
    with pytest.raises(ValueError):
        uoti.tse_remainder(y, 2, 1)


# ---- the generic remainder (the paper's Eq. 36) ----------------------------------

@pytest.mark.parametrize("n", [0, 1, 2])
def test_generic_remainder_matches_the_specific_ones(n):
    """tq.remainder(quantity, ...) is expectation_remainder,
    central_moment_remainder and sobol_indices_remainder, sensitivities
    included."""
    _, f, names, a, b = REM_CASES[1][:5]
    tq, y, extra = build(f, names, a, b, n + 1, param=REM_PARAMS)
    pairs = [(tq.remainder(tq.expectation, y, n, 1), tq.expectation_remainder(y, n, 1))]
    for k in (2, 3, 4):
        pairs.append((tq.remainder(tq.central_moment, y, n, 1, k), tq.central_moment_remainder(y, n, 1, k)))
    if n > 0:  # Sobol' indices of the constant T_0 are undefined
        S, V, _ = tq.remainder(tq.sobol_indices, y, n, 1)
        S2, V2 = tq.sobol_indices_remainder(y, n, 1)
        for r in range(1, len(S)):
            pairs += list(zip(S[r], S2[r])) + list(zip(V[r], V2[r]))
    for g, h in pairs:
        for basis in [None] + extra:
            u = real(g) if basis is None else (oti_deriv(g, [basis]) or 0.0)
            v = real(h) if basis is None else (oti_deriv(h, [basis]) or 0.0)
            assert u == pytest.approx(v, rel=1e-13, abs=1e-14)


@pytest.mark.parametrize("case", SOBOL_REM_CASES, ids=[c[0] for c in SOBOL_REM_CASES])
def test_total_sobol_and_shapley_remainders(case):
    """Remainders of the total Sobol' indices and Shapley values, which have
    no dedicated function, against the oracle: values and both parameter
    sensitivities at once."""
    _, f, names, a, b, n, l = case
    tq, y, extra = build(f, names, a, b, n + l, param=SOBOL_REM_PARAMS)
    ST = tq.remainder(tq.total_sobol_indices, y, n, l)
    SH = tq.remainder(tq.shapley_values, y, n, l)
    for basis, p in [(None, None)] + list(zip(extra, SOBOL_REM_PARAMS)):
        hi, lo = _oracle(case, n + l, p), _oracle(case, n, p)
        Vh, Vl = hi.central_moment(2), lo.central_moment(2)
        for i in range(len(names)):
            _check(ST[i], hi, lo, hi.total_sobol_indices()[i], lo.total_sobol_indices()[i],
                   basis, f"S_T{i + 1} {p or 'value'}")
            _check(SH[0][i], hi, lo, hi.shapley_effects()[i] * Vh, lo.shapley_effects()[i] * Vl,
                   basis, f"Shapley {i + 1} {p or 'value'}")


def test_generic_remainder_rejects_bad_arguments():
    _, f, names, a, b = TSE_CASE
    tq, y, _ = build(f, names, a, b, 2)
    with pytest.raises(ValueError):
        tq.remainder(tq.expectation, y, 1, 0)
    with pytest.raises(ValueError):
        tq.remainder(tq.expectation, y, 2, 1)

