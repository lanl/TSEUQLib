"""Plain (p fixed) Sobol' and Shapley values from the library against the
exact oracle: the finite-difference check relies on these being right.

    conda run -n pyoti python -B -I tests/verify_sobol_values.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itertools import combinations
import numpy as np, sympy as sp
from library_adapter import build, real, moti
from oracle import TSEOracle

def f3(x, m):
    return x[0]**3 + x[0] * x[1]**2 + x[1] * x[2] + m.exp(x[2] / 3)

names, a, b, n = ["N", "N", "U"], [1.0, 2.0, 0.5], [0.5, 0.3, 1.5], 2
orc = TSEOracle(lambda x: f3(x, sp), names, a, b, n)
tq, y, _ = build(f3, names, a, b, n, module=moti)
sob, V, _ = tq.sobol_indices(y)
S_or = orc.sobol_indices(); V_or, _ = orc.partial_variances()
print("total variance: lib", real(V[0][0]), " oracle", orc.value(orc.central_moment(2)))
for r in range(1, 4):
    for j, u in enumerate(combinations(range(3), r)):
        print(f"S{''.join(str(i+1) for i in u):4s} lib {real(sob[r][j]): .12f}  oracle {orc.value(S_or[u]): .12f}"
              f"   V lib {real(V[r][j]): .10g}  oracle {orc.value(V_or[u]): .10g}")
shap = np.asarray(tq.shapley_values_v2(y), dtype=float).ravel()
phi = [orc.value(v) for v in orc.shapley_effects()]
Vt = orc.value(orc.central_moment(2))
print("Shapley lib     :", shap)
print("oracle (normed) :", np.array(phi))
print("oracle * V      :", np.array(phi) * Vt)
print("total Sobol' (oracle only, no library function):", [round(orc.value(v), 12) for v in orc.total_sobol_indices()])
