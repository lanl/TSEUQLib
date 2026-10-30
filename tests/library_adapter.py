"""Run TSEUQLib on a test case: plain values, OTI parameter sensitivities, and
central finite differences of the plain values."""

import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyoti.sparse as oti  # noqa: E402

import tseuqlib.oti_moments as moti  # noqa: E402
import tseuqlib.oti_moments_sens as moti_sens  # noqa: E402
import tseuqlib.oti_util as uoti  # noqa: E402
import tseuqlib.rv_moments as rv  # noqa: E402


def build(f, names, a, b, n, param=None, kmax=None, module=moti_sens):
    """Build the TSE and the tse_uq object.

    The parameter (which, i) gets the extra OTI basis d + 1. The mean comes
    from rv_central_moments, because rv_mean_from_ab calls np.sqrt and fails
    on OTI numbers.
    """
    d = len(names)
    a = np.array(list(a), dtype=object)
    b = np.array(list(b), dtype=object)
    extra = []
    if param is not None:
        which, i = param
        tgt = a if which == "a" else b
        tgt[i] = tgt[i] + oti.e(d + 1, order=1)
        extra = [d + 1]
    with contextlib.redirect_stdout(io.StringIO()):
        mom = rv.rv_central_moments([np.array(names), a, b], kmax or 4 * n)
        mom.compute_central_moments()
    x = [mom.rv_mean[i] + oti.e(i + 1, order=n) for i in range(d)]
    y = f(x, oti)
    joint = uoti.build_rv_joint_moments(mom.rv_mu)
    return module.tse_uq(d, joint, extra_bases=extra), y, extra


def real(v):
    return float(v.real) if hasattr(v, "real") else float(v)


def oti_deriv(v, extra):
    """d/dp from the extra basis, or None if the result carries no OTI part."""
    if not hasattr(v, "get_deriv"):
        return None
    return float(v.get_deriv(extra))


def central_fd(quantity, f, names, a, b, n, param, h_rel=1e-5, **kw):
    """Central difference of a plain library quantity in parameter (which, i)."""
    which, i = param
    vals = []
    for s in (+1, -1):
        aa, bb = list(a), list(b)
        tgt = aa if which == "a" else bb
        h = h_rel * max(abs(tgt[i]), 1.0)
        tgt[i] = tgt[i] + s * h
        tq, y, _ = build(f, names, aa, bb, n, param=None, module=moti, **kw)
        vals.append(real(quantity(tq, y)))
    return (vals[0] - vals[1]) / (2 * h)


def _taylor_coefficients(y, d, n):
    """{alpha: c_alpha} of a TSE held as an OTI number in bases 1..d."""
    from itertools import combinations_with_replacement
    from math import factorial
    coeffs = {(0,) * d: real(y)}
    for k in range(1, n + 1):
        for dirs in combinations_with_replacement(range(1, d + 1), k):
            alpha = tuple(dirs.count(i) for i in range(1, d + 1))
            denom = 1
            for ai in alpha:
                denom *= factorial(ai)
            coeffs[alpha] = float(y.get_deriv(list(dirs))) / denom
    return coeffs


def central_fd_fixed(quantity, f, names, a, b, n, param, h_rel=1e-5, **kw):
    """Central difference for target (C): the surrogate is built once at p0
    and only the input distribution moves.

    For p0 +/- h the fixed polynomial sum c_alpha (x - m0)^alpha is re-centered
    at the new mean m(p) by substituting e_i -> e_i + (m_i(p) - m_i(p0)) in
    plain OTI arithmetic, and the plain moments are taken with the new input
    moments. Independent of the sympy oracle and of the extra-basis path.
    """
    d = len(names)
    _, y0, _ = build(f, names, a, b, n, param=None, module=moti, **kw)
    coeffs = _taylor_coefficients(y0, d, n)
    with contextlib.redirect_stdout(io.StringIO()):
        m0 = rv.rv_central_moments([np.array(names), np.array(a, dtype=object),
                                    np.array(b, dtype=object)], kw.get("kmax") or 4 * n)
        m0.compute_central_moments()
    which, i = param
    vals = []
    for s in (+1, -1):
        aa, bb = list(a), list(b)
        tgt = aa if which == "a" else bb
        h = h_rel * max(abs(tgt[i]), 1.0)
        tgt[i] = tgt[i] + s * h
        with contextlib.redirect_stdout(io.StringIO()):
            mom = rv.rv_central_moments([np.array(names), np.array(aa, dtype=object),
                                         np.array(bb, dtype=object)], kw.get("kmax") or 4 * n)
            mom.compute_central_moments()
        shifted = [oti.e(j + 1, order=n) + (mom.rv_mean[j] - m0.rv_mean[j]) for j in range(d)]
        Y = 0 * oti.e(1, order=n)
        for alpha, c in coeffs.items():
            term = c + 0 * oti.e(1, order=n)
            for j, aj in enumerate(alpha):
                if aj:
                    term = term * shifted[j]**aj
            Y = Y + term
        tq = moti.tse_uq(d, uoti.build_rv_joint_moments(mom.rv_mu))
        vals.append(real(quantity(tq, Y)))
    return (vals[0] - vals[1]) / (2 * h)
