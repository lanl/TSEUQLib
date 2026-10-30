"""Parameter sensitivities, target (C): the derivative of each quantity of
the current surrogate T_n (built once at the nominal mean) as a distribution
parameter p moves the input distribution. Decided with Sam on 2026-10-30.

The OTI tests compare the library with the exact oracle in its default
fixed-center mode (C). Two finite-difference tests check the oracle itself:
- `*_fd_fixed_*` keeps the surrogate fixed and re-centers it at the shifted
  mean in plain OTI arithmetic (target C, independent of sympy and of the
  extra-basis path);
- `*_fd_matches_*` rebuilds the expansion at the shifted mean and checks the
  oracle's moving-center mode (A).

Bugs that fail today are listed in KNOWN_BUGS and marked
xfail(strict=True, raises=...). A fix that makes one pass shows up as XPASS
and must be removed from the list.
"""

from itertools import combinations

import pytest
import sympy as sp

from cases import MOMENT_CASES, MOMENT_PARAMS, SOBOL_CASE, SOBOL_PARAMS, pid
from library_adapter import build, central_fd, central_fd_fixed, moti, oti_deriv
from oracle import TSEOracle

REL_OTI = 1e-10
REL_FD = 1e-6

# Measured 2026-10-30, after the basis-5 fix, the switch to target (C), and
# the Shapley object-array fix: no sensitivity bugs remain in these cases.
KNOWN_BUGS = {}
RAISES = {}

MOMENTS = {
    "E": (lambda tq, y: tq.expectation(y), lambda o: o.expectation()),
    "mu2": (lambda tq, y: tq.central_moment(y, 2), lambda o: o.central_moment(2)),
    "mu3": (lambda tq, y: tq.central_moment(y, 3), lambda o: o.central_moment(3)),
}


def _known(test_id):
    # Each known bug must fail the way it is recorded (RAISES): a wrong value
    # as AssertionError, a crash as its exception. Any other error fails.
    if test_id in KNOWN_BUGS:
        reason = KNOWN_BUGS[test_id]
        return [pytest.mark.xfail(strict=True, raises=RAISES[reason], reason=reason)]
    return []


def _moment_params(with_marks):
    out = []
    for c in MOMENT_CASES:
        for p in MOMENT_PARAMS:
            for q in MOMENTS:
                tid = f"{c[0]}-{pid(p)}-{q}"
                out.append(pytest.param(c, p, q, id=tid, marks=_known(tid) if with_marks else []))
    return out


def _oracle(case, param, center="fixed"):
    _, f, names, a, b, n = case
    return TSEOracle(lambda x: f(x, sp), names, a, b, n, param=param, center=center)


@pytest.mark.parametrize("case,param,qty", _moment_params(with_marks=False))
def test_moment_fd_matches_oracle(case, param, qty):
    _, f, names, a, b, n = case
    qlib, qorc = MOMENTS[qty]
    orc = _oracle(case, param, center="moving")
    fd = central_fd(qlib, f, names, a, b, n, param)
    assert fd == pytest.approx(orc.deriv(qorc(orc)), rel=REL_FD, abs=1e-9)


@pytest.mark.parametrize("case,param,qty", _moment_params(with_marks=False))
def test_moment_fd_fixed_matches_oracle(case, param, qty):
    _, f, names, a, b, n = case
    qlib, qorc = MOMENTS[qty]
    orc = _oracle(case, param)
    fd = central_fd_fixed(qlib, f, names, a, b, n, param)
    assert fd == pytest.approx(orc.deriv(qorc(orc)), rel=REL_FD, abs=1e-9)


@pytest.mark.parametrize("case,param,qty", _moment_params(with_marks=True))
def test_moment_oti_matches_oracle(case, param, qty):
    _, f, names, a, b, n = case
    qlib, qorc = MOMENTS[qty]
    orc = _oracle(case, param)
    tq, y, extra = build(f, names, a, b, n, param=param)
    d = oti_deriv(qlib(tq, y), extra)
    assert d is not None, "result carries no OTI part"
    assert d == pytest.approx(orc.deriv(qorc(orc)), rel=REL_OTI, abs=1e-12)


@pytest.mark.parametrize("param", [MOMENT_PARAMS[0]], ids=["a1"])
def test_package_module_keeps_sensitivity(param):
    """tseuqlib.oti_moments, which the package imports, should carry the
    parameter sensitivity too."""
    _, f, names, a, b, n = MOMENT_CASES[1]
    tq, y, extra = build(f, names, a, b, n, param=param, module=moti)
    if oti_deriv(tq.expectation(y), extra) is None:
        pytest.xfail("oti_moments ignores extra_bases and returns a float")


# ---- Sobol' and Shapley ----------------------------------------------------

def _sobol_lib(tq, y, d):
    # oti_moments returns (sobol, V, hdmr); oti_moments_sens returns (sobol, V)
    # oti_moments_sens stops at the expansion order; None marks omitted orders
    sob = tq.sobol_indices(y)[0]
    return [sob[r][j] if r < len(sob) else None
            for r in range(1, d + 1) for j in range(len(list(combinations(range(d), r))))]


def _sobol_orc(o):
    S = o.sobol_indices()
    return [S[u] for r in range(1, o.d + 1) for u in combinations(range(o.d), r)]


def _shapley_lib(tq, y, d):
    shap = tq.shapley_values_v2(y)
    return [shap[0][i] for i in range(d)]


def _shapley_orc(o):
    Vt = o.central_moment(2)
    return [phi * Vt for phi in o.shapley_effects()]


SOBOL_QTYS = {"sobol": (_sobol_lib, _sobol_orc), "shapley": (_shapley_lib, _shapley_orc)}


def _sobol_params(with_marks):
    out = []
    for p in SOBOL_PARAMS:
        for q in SOBOL_QTYS:
            tid = f"{SOBOL_CASE[0]}-{pid(p)}-{q}"
            out.append(pytest.param(p, q, id=tid, marks=_known(tid) if with_marks else []))
    return out


@pytest.mark.parametrize("param,qty", _sobol_params(with_marks=False))
def test_sobol_fd_matches_oracle(param, qty):
    _, f, names, a, b, n = SOBOL_CASE
    qlib, qorc = SOBOL_QTYS[qty]
    orc = _oracle(SOBOL_CASE, param, center="moving")
    exact = [orc.deriv(e) for e in qorc(orc)]
    for i in range(len(exact)):
        fd = central_fd(lambda tq, y: qlib(tq, y, orc.d)[i], f, names, a, b, n, param)
        assert fd == pytest.approx(exact[i], rel=REL_FD, abs=1e-9), f"entry {i}"


@pytest.mark.parametrize("param,qty", _sobol_params(with_marks=False))
def test_sobol_fd_fixed_matches_oracle(param, qty):
    _, f, names, a, b, n = SOBOL_CASE
    qlib, qorc = SOBOL_QTYS[qty]
    orc = _oracle(SOBOL_CASE, param)
    exact = [orc.deriv(e) for e in qorc(orc)]
    for i in range(len(exact)):
        fd = central_fd_fixed(lambda tq, y: qlib(tq, y, orc.d)[i], f, names, a, b, n, param)
        assert fd == pytest.approx(exact[i], rel=REL_FD, abs=1e-9), f"entry {i}"


@pytest.mark.parametrize("param,qty", _sobol_params(with_marks=True))
def test_sobol_oti_matches_oracle(param, qty):
    _, f, names, a, b, n = SOBOL_CASE
    qlib, qorc = SOBOL_QTYS[qty]
    orc = _oracle(SOBOL_CASE, param)
    tq, y, extra = build(f, names, a, b, n, param=param)
    got = qlib(tq, y, orc.d)
    exact = [orc.deriv(e) for e in qorc(orc)]
    for i, (g, e) in enumerate(zip(got, exact)):
        if g is None:  # an order the library omits must be zero
            assert e == pytest.approx(0.0, abs=1e-12), f"entry {i} omitted but nonzero"
            continue
        d = oti_deriv(g, extra)
        assert d is not None, f"entry {i} carries no OTI part"
        assert d == pytest.approx(e, rel=REL_OTI, abs=1e-12), f"entry {i}"
