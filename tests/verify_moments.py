"""Print value and sensitivity errors of E and central moments against the
exact oracle (target C) and fixed-surrogate finite differences. Run:

    conda run -n pyoti python -B -I tests/verify_moments.py
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sympy as sp  # noqa: E402

from library_adapter import build, central_fd_fixed, oti_deriv, real  # noqa: E402
from oracle import TSEOracle  # noqa: E402


def f_cubic(x, m):
    # degree 3 with an interaction: above n = 2, so truncated terms matter
    return x[0]**3 + x[0] * x[1]**2 + 2 * x[1]


def f_exp(x, m):
    return m.exp(x[0] / 2) * x[1] + x[1]**3


CASES = [
    # (label, f, names, a, b, n)
    ("N cubic n=2", f_cubic, ["N", "N"], [1.0, 2.0], [0.5, 0.3], 2),
    ("N cubic n=3", f_cubic, ["N", "N"], [1.0, 2.0], [0.5, 0.3], 3),
    ("U cubic n=2", f_cubic, ["U", "U"], [0.5, 1.5], [1.5, 2.5], 2),
    ("N exp   n=2", f_exp, ["N", "N"], [1.0, 2.0], [0.5, 0.3], 2),
]

QUANTITIES = [
    ("E", lambda tq, y: tq.expectation(y), lambda o: o.expectation()),
    ("mu2", lambda tq, y: tq.central_moment(y, 2), lambda o: o.central_moment(2)),
    ("mu3", lambda tq, y: tq.central_moment(y, 3), lambda o: o.central_moment(3)),
]


def rel(x, ref):
    return abs(x - ref) / max(abs(ref), 1e-300)


def main():
    print(f"{'case':14s} {'param':6s} {'qty':4s} {'value err':>10s} {'exact d/dp':>14s} "
          f"{'OTI d/dp':>14s} {'OTI err':>9s} {'FD err':>9s}")
    for label, f, names, a, b, n in CASES:
        for param in (("a", 0), ("b", 0), ("b", 1)):
            orc = TSEOracle(lambda x: f(x, sp), names, a, b, n, param=param)
            for qname, qlib, qorc in QUANTITIES:
                expr = qorc(orc)
                exact_v, exact_d = orc.value(expr), orc.deriv(expr)
                pname = f"{param[0]}{param[1] + 1}"
                try:
                    tq, y, extra = build(f, names, a, b, n, param=param)
                    q = qlib(tq, y)
                    v_err = rel(real(q), exact_v)
                    d = oti_deriv(q, extra)
                    d_txt = "no OTI part" if d is None else f"{d:14.8g}"
                    d_err = "-" if d is None else f"{rel(d, exact_d):9.1e}"
                except Exception as e:
                    v_err, d_txt, d_err = float("nan"), type(e).__name__, "crash"
                try:
                    fd = central_fd_fixed(qlib, f, names, a, b, n, param)
                    fd_err = f"{rel(fd, exact_d):9.1e}"
                except Exception as e:
                    fd_err = type(e).__name__[:9]
                print(f"{label:14s} {pname:6s} {qname:4s} {v_err:10.1e} {exact_d:14.8g} "
                      f"{d_txt:>14s} {d_err:>9s} {fd_err:>9s}")


if __name__ == "__main__":
    main()
