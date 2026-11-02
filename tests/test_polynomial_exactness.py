"""For a polynomial model of degree p, every quantity of a Taylor series of
order n >= p equals the model's own exact value (the claim of
examples/plots/polynomial_exactness.png). One random cubic on two inputs per
distribution, against the reference of
examples/verification/polynomial_exactness.py: the polynomial integrated
against exact input moments at 50 digits, no Taylor series involved."""

import os
import random
import sys

import pytest
from mpmath import mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "examples", "verification"))
import polynomial_exactness as pe  # noqa: E402

P, D = 3, 2
CASES = [("N", [1.2, 0.8], [0.3, 0.2]), ("U", [0.2, 0.5], [1.1, 1.7]),
         ("T", [0.1, 0.6], [1.0, 1.4]), ("K", [2.0, 3.0], [2.5, 1.8]),
         ("LN", [0.0, -0.3], [0.3, 0.2])]


@pytest.mark.parametrize("n", [P, P + 1])
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_polynomial_quantities_are_exact(case, n):
    code, a, b = case
    exact_mean, mom = pe.input_moments(code, a, b, 4 * P)
    m = [float(x) for x in exact_mean]
    s = [float(mp.sqrt(mom[i][2])) for i in range(D)]
    coeffs, f = pe.make_polynomial(random.Random(7), D, P, m, s)
    ref = pe.reference(coeffs, D, P, m, s, exact_mean, mom)
    lib = pe.library(f, code, D, a, b, n)
    for key, ex in ref.items():
        assert lib[key] == pytest.approx(float(ex), rel=1e-10, abs=1e-12), pe.name_of(key)
