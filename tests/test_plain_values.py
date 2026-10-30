"""Plain values (no parameter sensitivity) against the exact oracle. These
pass today and guard what the library already gets right."""

from itertools import combinations

import numpy as np
import pytest
import sympy as sp

from cases import MOMENT_CASES, SOBOL_CASE
from library_adapter import build, moti, moti_sens, real
from oracle import TSEOracle

REL = 1e-12


@pytest.mark.parametrize("module", [moti, moti_sens], ids=["oti_moments", "oti_moments_sens"])
@pytest.mark.parametrize("case", MOMENT_CASES, ids=[c[0] for c in MOMENT_CASES])
def test_moments(case, module):
    _, f, names, a, b, n = case
    orc = TSEOracle(lambda x: f(x, sp), names, a, b, n)
    tq, y, _ = build(f, names, a, b, n, module=module)
    assert real(tq.expectation(y)) == pytest.approx(orc.value(orc.expectation()), rel=REL)
    for k in (2, 3, 4):
        assert real(tq.central_moment(y, k)) == pytest.approx(
            orc.value(orc.central_moment(k)), rel=REL, abs=1e-12)


@pytest.fixture(scope="module")
def sobol_pair():
    _, f, names, a, b, n = SOBOL_CASE
    orc = TSEOracle(lambda x: f(x, sp), names, a, b, n)
    tq, y, _ = build(f, names, a, b, n, module=moti)
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
