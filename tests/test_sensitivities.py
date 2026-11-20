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

Several parameters at once (one OTI basis each) are checked against the
single-parameter oracles; extra_bases is optional (every basis above n_dim is
a parameter).

Bugs that fail today are listed in KNOWN_BUGS and marked
xfail(strict=True, raises=...). A fix that makes one pass shows up as XPASS
and must be removed from the list.
"""

from itertools import combinations

import pytest
import sympy as sp

from cases import MOMENT_CASES, MOMENT_PARAMS, SOBOL_CASES, pid
from library_adapter import build, central_fd, central_fd_fixed, moti, moti_sens, oti_deriv
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
    "mu4": (lambda tq, y: tq.central_moment(y, 4), lambda o: o.central_moment(4)),
}


def _known(test_id):
    # Each known bug must fail the way it is recorded (RAISES): a wrong value
    # as AssertionError, a crash as its exception. Any other error fails.
    if test_id in KNOWN_BUGS:
        reason = KNOWN_BUGS[test_id]
        return [pytest.mark.xfail(strict=True, raises=RAISES[reason], reason=reason)]
    return []


def _slow_oracle(case):
    # The moving-center checks verify the oracle's mode (A), not the library,
    # and its sympy work is slow for Kumaraswamy and lognormal inputs; the N,
    # U and T cases cover mode (A). K and LN cases keep the oracle (C) and
    # fixed-surrogate checks.
    return any(name in ("K", "LN") for name in case[2])


def _moment_params(with_marks, moving=False):
    out = []
    for c in MOMENT_CASES:
        if moving and _slow_oracle(c):
            continue
        for p in MOMENT_PARAMS:
            for q in MOMENTS:
                tid = f"{c[0]}-{pid(p)}-{q}"
                out.append(pytest.param(c, p, q, id=tid, marks=_known(tid) if with_marks else []))
    return out


_ORACLES = {}


def _oracle(case, param, center="fixed"):
    # cached: the n = 3 oracles take a few seconds of sympy each
    key = (case[0], param, center)
    if key not in _ORACLES:
        _, f, names, a, b, n = case
        _ORACLES[key] = TSEOracle(lambda x: f(x, sp), names, a, b, n, param=param, center=center)
    return _ORACLES[key]


@pytest.mark.parametrize("case,param,qty", _moment_params(with_marks=False, moving=True))
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


def test_package_and_alias_share_one_implementation():
    """The package, oti_moments and the oti_moments_sens alias share one
    tse_uq (merged 2026-11-01), so these tests cover what `import tseuqlib`
    gives."""
    import tseuqlib
    assert tseuqlib.tse_uq is moti.tse_uq
    assert moti_sens.tse_uq is moti.tse_uq


# ---- partial variances, Sobol' and Shapley ---------------------------------

def _subsets(d):
    # oracle order; the library is read by label (its list order differs for d >= 4)
    return [tuple(i + 1 for i in u) for r in range(1, d + 1) for u in combinations(range(d), r)]


def _V_lib(tq, y, d):
    # orders above n are padded with 0
    V = tq.sobol_indices_by_term(y)[1]
    return [V[u] for u in _subsets(d)]


def _V_orc(o):
    V, _ = o.partial_variances()
    return [V[u] for r in range(1, o.d + 1) for u in combinations(range(o.d), r)]


def _sobol_lib(tq, y, d):
    S = tq.sobol_indices_by_term(y)[0]
    return [S[u] for u in _subsets(d)]


def _sobol_orc(o):
    S = o.sobol_indices()
    return [S[u] for r in range(1, o.d + 1) for u in combinations(range(o.d), r)]


def _shapley_lib(tq, y, d):
    shap = tq.shapley_values_v2(y)
    return [shap[0][i] for i in range(d)]


def _shapley_orc(o):
    Vt = o.central_moment(2)
    return [phi * Vt for phi in o.shapley_effects()]


def _total_lib(tq, y, d):
    return list(tq.total_sobol_indices(y))


def _total_orc(o):
    return o.total_sobol_indices()


SOBOL_QTYS = {"V": (_V_lib, _V_orc), "sobol": (_sobol_lib, _sobol_orc),
              "shapley": (_shapley_lib, _shapley_orc), "total": (_total_lib, _total_orc)}


def _sobol_params(with_marks, moving=False):
    out = []
    for case, params in SOBOL_CASES:
        if moving and _slow_oracle(case):
            continue
        for p in params:
            for q in SOBOL_QTYS:
                tid = f"{case[0]}-{pid(p)}-{q}"
                out.append(pytest.param(case, p, q, id=tid, marks=_known(tid) if with_marks else []))
    return out


def _assert_entry(g, e, basis, msg):
    """One library entry against the oracle derivative for one basis."""
    if not hasattr(g, "get_deriv"):  # a padded order above n must be exactly zero
        assert g == 0, f"{msg}: a plain nonzero number"
        assert e == pytest.approx(0.0, abs=1e-12), f"{msg}: padded but nonzero"
        return
    assert oti_deriv(g, [basis]) == pytest.approx(e, rel=REL_OTI, abs=1e-12), msg


@pytest.mark.parametrize("case,param,qty", _sobol_params(with_marks=False, moving=True))
def test_sobol_fd_matches_oracle(case, param, qty):
    _, f, names, a, b, n = case
    qlib, qorc = SOBOL_QTYS[qty]
    orc = _oracle(case, param, center="moving")
    exact = [orc.deriv(e) for e in qorc(orc)]
    for i in range(len(exact)):
        fd = central_fd(lambda tq, y: qlib(tq, y, orc.d)[i], f, names, a, b, n, param)
        assert fd == pytest.approx(exact[i], rel=REL_FD, abs=1e-9), f"entry {i}"


@pytest.mark.parametrize("case,param,qty", _sobol_params(with_marks=False))
def test_sobol_fd_fixed_matches_oracle(case, param, qty):
    _, f, names, a, b, n = case
    qlib, qorc = SOBOL_QTYS[qty]
    orc = _oracle(case, param)
    exact = [orc.deriv(e) for e in qorc(orc)]
    for i in range(len(exact)):
        fd = central_fd_fixed(lambda tq, y: qlib(tq, y, orc.d)[i], f, names, a, b, n, param)
        assert fd == pytest.approx(exact[i], rel=REL_FD, abs=1e-9), f"entry {i}"


@pytest.mark.parametrize("case,param,qty", _sobol_params(with_marks=True))
def test_sobol_oti_matches_oracle(case, param, qty):
    _, f, names, a, b, n = case
    qlib, qorc = SOBOL_QTYS[qty]
    orc = _oracle(case, param)
    tq, y, extra = build(f, names, a, b, n, param=param)
    got = qlib(tq, y, orc.d)
    exact = [orc.deriv(e) for e in qorc(orc)]
    for i, (g, e) in enumerate(zip(got, exact)):
        _assert_entry(g, e, extra[0], f"entry {i}")


# ---- several parameters at once ---------------------------------------------
# Each parameter on its own OTI basis in one run, each derivative against its
# single-parameter oracle. Sized within pyoti's basis limits (inputs plus
# parameters <= 5 at order 12, the order mu4 needs at n = 3).

@pytest.mark.parametrize("case", MOMENT_CASES, ids=[c[0] for c in MOMENT_CASES])
def test_moments_all_params_at_once(case):
    _, f, names, a, b, n = case
    tq, y, extra = build(f, names, a, b, n, param=MOMENT_PARAMS)
    for q, (qlib, qorc) in MOMENTS.items():
        got = qlib(tq, y)
        for basis, p in zip(extra, MOMENT_PARAMS):
            orc = _oracle(case, p)
            assert oti_deriv(got, [basis]) == pytest.approx(
                orc.deriv(qorc(orc)), rel=REL_OTI, abs=1e-12), f"{q} {pid(p)}"


@pytest.mark.parametrize("case,params", SOBOL_CASES, ids=[c[0] for c, _ in SOBOL_CASES])
def test_sobol_all_params_at_once(case, params):
    _, f, names, a, b, n = case
    tq, y, extra = build(f, names, a, b, n, param=params)
    for q, (qlib, qorc) in SOBOL_QTYS.items():
        got = qlib(tq, y, len(names))
        for basis, p in zip(extra, params):
            orc = _oracle(case, p)
            for i, (g, e) in enumerate(zip(got, qorc(orc))):
                _assert_entry(g, orc.deriv(e), basis, f"{q} entry {i} {pid(p)}")


def test_extra_bases_optional():
    """Every basis above n_dim is a parameter, so leaving extra_bases out
    gives the same results."""
    case, params = SOBOL_CASES[1]
    _, f, names, a, b, n = case
    d = len(names)
    with_extra = build(f, names, a, b, n, param=params)
    without = build(f, names, a, b, n, param=params, pass_extra=False)
    vals = []
    for tq, y, extra in (with_extra, without):
        out = [tq.expectation(y), tq.central_moment(y, 2), tq.central_moment(y, 3)]
        out += _V_lib(tq, y, d) + _sobol_lib(tq, y, d) + _shapley_lib(tq, y, d)
        vals.append([[oti_deriv(v, [bs]) if hasattr(v, "get_deriv") else 0.0 for bs in extra] for v in out])
    assert vals[0] == vals[1]
