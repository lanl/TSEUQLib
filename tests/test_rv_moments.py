"""Input central moments against exact SymPy values, up to order 20, where
cancellation shows: Kumaraswamy and lognormal, each value with its
derivatives in both distribution parameters at once (one OTI basis each).
Every sensitivity starts here. Until 2026-11-02 the Kumaraswamy moments came
from a double-precision alternating sum: 1e-10 relative error by order 12,
1e-5 by order 20."""

import contextlib
import io

import numpy as np
import pytest
import sympy as sp

from library_adapter import oti, rv
from oracle import _rv_mean_and_central_moments

KMAX = 20
CASES = [("K", 2.0, 2.5), ("K", 3.5, 1.5), ("K", 3.0, 3.0), ("K", 1.5, 3.5),
         ("LN", 0.0, 0.3), ("LN", -0.3, 0.1)]
A, B = sp.symbols("a b", positive=True)


def _library(name, a, b):
    pa = np.array([a + oti.e(1, order=1)], dtype=object)
    pb = np.array([b + oti.e(2, order=1)], dtype=object)
    with contextlib.redirect_stdout(io.StringIO()):
        m = rv.rv_central_moments([np.array([name]), pa, pb], KMAX)
        m.compute_central_moments()
    return [mk[0] for mk in m.rv_mu]


@pytest.mark.parametrize("case", CASES, ids=[f"{n}-{a}-{b}" for n, a, b in CASES])
def test_central_moments_and_their_parameter_derivatives(case):
    name, a, b = case
    got = _library(name, a, b)
    _, exact = _rv_mean_and_central_moments(name, A, B, KMAX)
    at = {A: sp.Rational(str(a)), B: sp.Rational(str(b))}
    for k in range(2, KMAX + 1):
        e = exact[k - 1]
        want = [float(sp.N(x.subs(at), 30)) for x in (e, sp.diff(e, A), sp.diff(e, B))]
        have = [float(got[k - 1].real), float(got[k - 1].get_deriv([1])),
                float(got[k - 1].get_deriv([2]))]
        for what, h, w in zip(("value", "d/da", "d/db"), have, want):
            assert h == pytest.approx(w, rel=1e-12, abs=1e-12 * abs(want[0])), f"mu{k} {what}"
