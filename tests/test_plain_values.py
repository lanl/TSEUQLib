"""Plain values (no parameter sensitivity) against the exact oracle. These
pass today and guard what the library already gets right."""

from itertools import combinations

import numpy as np
import pytest
import sympy as sp

from cases import MOMENT_CASES, SOBOL_CASE
from library_adapter import build, real
from oracle import TSEOracle

REL = 1e-12


@pytest.mark.parametrize("case", MOMENT_CASES, ids=[c[0] for c in MOMENT_CASES])
def test_moments(case):
    _, f, names, a, b, n = case
    orc = TSEOracle(lambda x: f(x, sp), names, a, b, n)
    tq, y, _ = build(f, names, a, b, n)
    assert real(tq.expectation(y)) == pytest.approx(orc.value(orc.expectation()), rel=REL)
    for k in (2, 3, 4):
        assert real(tq.central_moment(y, k)) == pytest.approx(
            orc.value(orc.central_moment(k)), rel=REL, abs=1e-12)


@pytest.fixture(scope="module")
def sobol_pair():
    _, f, names, a, b, n = SOBOL_CASE
    orc = TSEOracle(lambda x: f(x, sp), names, a, b, n)
    tq, y, _ = build(f, names, a, b, n)
    return orc, tq, y


def test_sobol_and_partial_variances(sobol_pair):
    orc, tq, y = sobol_pair
    sob, V, _ = tq.sobol_indices(y)
    S_or = orc.sobol_indices()
    V_or, _ = orc.partial_variances()
    assert real(V[0][0]) == pytest.approx(orc.value(orc.central_moment(2)), rel=REL)
    for r in range(1, orc.d + 1):
        for j, u in enumerate(combinations(range(orc.d), r)):
            assert real(sob[r][j]) == pytest.approx(orc.value(S_or[u]), rel=REL, abs=1e-14)
            assert real(V[r][j]) == pytest.approx(orc.value(V_or[u]), rel=REL, abs=1e-14)


def test_shapley_v2_is_unnormalized(sobol_pair):
    """shapley_values_v2 returns phi_i * V (variance units), not phi_i."""
    orc, tq, y = sobol_pair
    shap = np.asarray(tq.shapley_values_v2(y), dtype=float).ravel()
    Vt = orc.value(orc.central_moment(2))
    expected = [orc.value(v) * Vt for v in orc.shapley_effects()]
    assert shap == pytest.approx(expected, rel=REL)


def test_sobol_orders_padded_to_dimension(sobol_pair):
    """Orders above n have zero partial variance and are padded up to d, so
    the result keeps the shape oti_moments always returned."""
    orc, tq, y = sobol_pair
    sob, V, hdmr = tq.sobol_indices(y)
    assert y.order < orc.d
    assert [len(s) for s in sob] == [len(list(combinations(range(orc.d), r))) for r in range(orc.d + 1)]
    for r in range(y.order + 1, orc.d + 1):
        assert sob[r] == [0] * len(sob[r]) and V[r] == [0] * len(V[r]) and hdmr[r] == [0] * len(hdmr[r])


def test_sobol_max_order(sobol_pair):
    """max_order limits the orders computed (oti_moments used to ignore it)."""
    orc, tq, y = sobol_pair
    sob_full, V_full, _ = tq.sobol_indices(y)
    sob, V, hdmr = tq.sobol_indices(y, max_order=1)
    assert len(sob) == len(V) == len(hdmr) == 2
    for j in range(orc.d):
        assert real(sob[1][j]) == pytest.approx(real(sob_full[1][j]), rel=REL)
        assert real(V[1][j]) == pytest.approx(real(V_full[1][j]), rel=REL)


def test_real_results_are_numpy_floats(sobol_pair):
    """Real results, padded orders included, are numpy floats, as before
    2026-11-01: ishigami_paper divides by an exactly vanishing partial
    variance and expects inf with a warning, not ZeroDivisionError."""
    orc, tq, y = sobol_pair
    sob, V, hdmr = tq.sobol_indices(y)
    assert isinstance(tq.expectation(y), np.floating)
    assert isinstance(tq.central_moment(y, 2), np.floating)
    for r in range(1, orc.d + 1):
        for v in V[r] + sob[r]:
            assert isinstance(v, np.floating), (r, type(v))
    with np.errstate(divide="ignore"):
        assert np.isinf(1.0 / V[orc.d][0])  # the padded order above n
