"""The paper's numbers, Section 3.1 (tseuqlib-7.pdf): f = (1 + 2x1 + 3x2 + 5x3 +
7x4)^3 with Kumaraswamy inputs (Table 2). The third-order expansion is f
itself, so its metrics are the exact ones of Table 3.

Table 3 prints S13 = 139975120954015875/801771070093938026, which is 56 times
V13/V; the correct value, used here, has denominator 44899179925260529456
(the printed relative error, 1.78E-15, is against the correct value).
Checked 2026-11-01."""

import contextlib
import io
from fractions import Fraction as F

import numpy as np
import pytest

from library_adapter import central_fd_fixed, moti, oti, rv, uoti

NAMES, A, B = ["K"] * 4, [1.0, 1.0, 2.0, 2.0], [3.0, 8.0, 6.0, 4.0]

TABLE3 = {
    "E": F(209949406, 675675), "V": F(400885535046969013, 7761123995625),
    "mu3": F(1664543546130905925590525911, 99635951659134515625),
    "mu4": F(31619003734529885887032538772484507689, 2064526663543428557910937500),
    "V1": F(914603454068, 326064375), "V2": F(17169878229461417, 10043807523750),
    "V3": F(2443557721082767, 198684774288), "V4": F(29265383540220103, 913073411250),
    "V12": F(743757176927, 33817533750), "V13": F(221786684023, 1377295920),
    "V14": F(4693801933757, 11272511250), "V23": F(3885725843299, 40175230095),
    "V24": F(114169056362554, 456536705625), "V34": F(486419013137, 265621356),
    "V123": F(958844, 3006003), "V124": F(13844, 16875),
    "V134": F(829639771, 135270135), "V234": F(13274236336, 3652293645),
    "S1": F(152388483706447956, 2806198745328783091), "S2": F(291887929900844089, 8819481771033318286),
    "S3": F(1527223575676729375, 6414168560751504208), "S4": F(497511520183741751, 801771070093938026),
    "S12": F(341384544209493, 801771070093938026), "S13": F(139975120954015875, 44899179925260529456),
    "S14": F(6463365262783389, 801771070093938026), "S23": F(8257167417010375, 4409740885516659143),
    "S24": F(1940873958163418, 400885535046969013), "S34": F(56850222160386875, 1603542140187876052),
    "S123": F(2475615352500, 400885535046969013), "S124": F(6367111146396, 400885535046969013),
    "S134": F(47600581861125, 400885535046969013), "S234": F(28207752214000, 400885535046969013),
    "phi1": F(35305143937534859, 11362691340000), "phi2": F(22845460896096389, 12052569028500),
    "phi3": F(6125547009070672973, 458961828605280), "phi4": F(72981376368499943, 2191376187000),
}


def f40(x, m):
    return (1 + 2 * x[0] + 3 * x[1] + 5 * x[2] + 7 * x[3])**3


def _tse(a, b, n, kmax=12):
    with contextlib.redirect_stdout(io.StringIO()):
        mom = rv.rv_central_moments([np.array(NAMES), np.array(a, dtype=object),
                                     np.array(b, dtype=object)], kmax)
        mom.compute_central_moments()
    x = [mom.rv_mean[i] + oti.e(i + 1, order=n) for i in range(4)]
    return moti.tse_uq(4, uoti.build_rv_joint_moments(mom.rv_mu)), f40(x, oti)


def metrics(tq, y):
    """Every Table 3 quantity, keyed as in the table (the library orders
    interaction terms by OTI direction; build_hdmr's term_struct names them)."""
    out = {"E": tq.expectation(y), "V": tq.central_moment(y, 2),
           "mu3": tq.central_moment(y, 3), "mu4": tq.central_moment(y, 4)}
    sob, V, _ = tq.sobol_indices(y)
    _, terms, _ = tq.build_hdmr(y)
    for r in range(1, 4):
        for j, t in enumerate(terms[r]):
            key = "".join(str(i) for i in sorted(t))
            out["V" + key], out["S" + key] = V[r][j], sob[r][j]
    shap = tq.shapley_values_v2(y)[0]
    for i in range(4):
        out[f"phi{i + 1}"] = shap[i]
    return out


@pytest.fixture(scope="module")
def table3_lib():
    return metrics(*_tse(A, B, 3))


def test_table3_consistent():
    """The table's own identities (with S13 corrected)."""
    V = TABLE3["V"]
    parts = [k for k in TABLE3 if k.startswith("V") and k != "V"]
    assert sum(TABLE3[k] for k in parts) == V  # V1234 = 0
    assert sum(TABLE3[f"phi{i}"] for i in range(1, 5)) == V
    for k in parts:
        assert TABLE3["S" + k[1:]] == TABLE3[k] / V, k


@pytest.mark.parametrize("key", list(TABLE3))
def test_table3(table3_lib, key):
    assert float(table3_lib[key]) == pytest.approx(float(TABLE3[key]), rel=1e-12)


def test_remainders_in_the_text():
    """Section 3.1: V[Y1] ~ 39024.5, eps1[V[Y1]] ~ 7406.5, eps1[V[Y2]] ~ 5222.0."""
    tq, y = _tse(A, B, 5, kmax=20)
    assert float(tq.central_moment(y.truncate_order(2), 2)) == pytest.approx(39024.5, abs=0.05)
    assert float(tq.central_moment_remainder(y, 1, 1, 2)) == pytest.approx(7406.5, abs=0.05)
    assert float(tq.central_moment_remainder(y, 2, 1, 2)) == pytest.approx(5222.0, abs=0.05)


TABLE4_ROWS = {
    # name: (remainder eps_m[xi[Y_n]], xi[Y_n]); V1, V12, V123 are the first
    # entries of their order
    "E": (lambda tq, y, n, m: tq.expectation_remainder(y, n, m),
          lambda tq, y, n: tq.expectation(y.truncate_order(n + 1))),
    "V": (lambda tq, y, n, m: tq.central_moment_remainder(y, n, m, 2),
          lambda tq, y, n: tq.central_moment(y.truncate_order(n + 1), 2)),
    "mu3": (lambda tq, y, n, m: tq.central_moment_remainder(y, n, m, 3),
            lambda tq, y, n: tq.central_moment(y.truncate_order(n + 1), 3)),
    "mu4": (lambda tq, y, n, m: tq.central_moment_remainder(y, n, m, 4),
            lambda tq, y, n: tq.central_moment(y.truncate_order(n + 1), 4)),
}
for _key, _r in (("V1", 1), ("V12", 2), ("V123", 3)):
    TABLE4_ROWS[_key] = (lambda tq, y, n, m, r=_r: tq.sobol_indices_remainder(y, n, m)[1][r][0],
                         lambda tq, y, n, r=_r: tq.sobol_indices(y.truncate_order(n + 1))[1][r][0]
                         if n > 0 else 0.0)


@pytest.mark.parametrize("key", list(TABLE4_ROWS))
def test_table4(key):
    """Table 4: once m + n >= 3 (the degree of f), the remainder eps_m[xi[Y_n]]
    equals the actual error xi[f] - xi[Y_n] (relative error ~1e-15); from
    n = 3 on, both are zero."""
    tq, y = _tse(A, B, 5, kmax=20)
    rem, val = TABLE4_ROWS[key]
    exact = float(TABLE3[key])
    for n in range(0, 3):
        actual = exact - float(val(tq, y, n))
        assert float(rem(tq, y, n, 3 - n)) == pytest.approx(actual, rel=1e-10, abs=1e-12 * abs(exact)), n
    for m in (1, 2):
        assert abs(float(rem(tq, y, 3, m))) <= 1e-12 * abs(exact), m


@pytest.mark.parametrize("param", [("a", 0), ("b", 0)], ids=["a1", "b1"])
def test_figure1_sensitivities(param):
    """Figure 1: d xi[Y3]/d a1 and d xi[Y3]/d b1 of every Table 3 quantity
    agree with finite differences (the paper: within 1e-6). Here central
    differences of the library's plain metrics, fixed surrogate (Y3 = f, so
    the expansion point does not matter)."""
    which, i = param
    a = np.array(A, dtype=object)
    b = np.array(B, dtype=object)
    (a if which == "a" else b)[i] = (A if which == "a" else B)[i] + oti.e(5, order=1)
    tq, y = _tse(a, b, 3)
    got = metrics(tq, y)
    keys = list(TABLE3)
    fd = {k: central_fd_fixed(lambda tq, y, k=k: metrics(tq, y)[k], f40, NAMES, A, B, 3, param, kmax=12)
          for k in keys}
    for k in keys:
        d = float(got[k].get_deriv([5])) if hasattr(got[k], "get_deriv") else 0.0
        assert d == pytest.approx(fd[k], rel=1e-6, abs=1e-9 * max(1.0, abs(float(TABLE3[k])))), k


# ---- Section 3.2: Ishigami, U(-pi, pi)^3, 1 <= n <= 20, remainder m = 2 -------

ISHIGAMI_EXACT = {  # Equations 43-48
    "E": 3.5, "V": 49 / 8 + 0.1 * np.pi**4 / 5 + 0.01 * np.pi**8 / 18 + 0.5,
    "V1": 0.1 * np.pi**4 / 5 + 0.01 * np.pi**8 / 50 + 0.5, "V2": 49 / 8,
    "V13": 8 * 0.01 * np.pi**8 / 225,
}


@pytest.fixture(scope="module")
def ishigami():
    """E, V, V1, V2, V13 of Y_n for n = 1..22 (expansion order 22 = 20 + m)."""
    nmax = 22
    with contextlib.redirect_stdout(io.StringIO()):
        mom = rv.rv_central_moments([np.array(["U"] * 3), np.array([-np.pi] * 3, dtype=object),
                                     np.array([np.pi] * 3, dtype=object)], 2 * nmax)
        mom.compute_central_moments()
    x = [mom.rv_mean[i] + oti.e(i + 1, order=nmax) for i in range(3)]
    y = oti.sin(x[0]) + 7 * oti.sin(x[1])**2 + 0.1 * x[2]**4 * oti.sin(x[0])
    tq = moti.tse_uq(3, uoti.build_rv_joint_moments(mom.rv_mu))
    out = {}
    for n in range(1, nmax + 1):
        yn = y.truncate_order(n + 1)
        sob, V, _ = tq.sobol_indices(yn)
        _, terms, _ = tq.build_hdmr(yn)
        pairs = [tuple(sorted(t)) for t in terms[2]] if len(terms) > 2 else []
        out[n] = {"E": float(tq.expectation(yn)), "V": float(V[0][0]), "V1": float(V[1][0]),
                  "V2": float(V[1][1]), "V13": float(V[2][pairs.index((1, 3))]) if (1, 3) in pairs else 0.0}
    return out


def test_ishigami_within_1_percent_at_18(ishigami):
    """Section 3.2: the metrics of Y_18 are within 1% of the exact ones."""
    for k, exact in ISHIGAMI_EXACT.items():
        assert abs(1 - ishigami[18][k] / exact) < 0.01, k


# |1 - eps_2[xi[Y_n]] / (xi[f] - xi[Y_n])| read from Figure 2c (log scale), so
# checked to 15%: enough to show the figure is reproduced.
FIGURE_2C = {
    3: {"E": 1.35, "V2": 4.2, "V1": 3.6, "V13": 5.6},
    5: {"E": 0.65, "V2": 0.53, "V1": 0.0085, "V13": 0.04},
    8: {"E": 0.27, "V2": 0.032, "V1": 2.9, "V13": 0.57},
    20: {"E": 0.066, "V2": 0.067, "V13": 0.024},
}


@pytest.mark.parametrize("n", FIGURE_2C)
def test_ishigami_figure_2c(ishigami, n):
    for k, read in FIGURE_2C[n].items():
        actual = ISHIGAMI_EXACT[k] - ishigami[n][k]
        predicted = ishigami[n + 2][k] - ishigami[n][k]
        assert abs(1 - predicted / actual) == pytest.approx(read, rel=0.15), k
