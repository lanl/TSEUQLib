"""Shared test cases. Each f takes (x, m), with m the math module: sympy for
the oracle, pyoti.sparse for the library. Every f has degree above the
expansion order or is non-polynomial, so truncated top-order terms matter."""


def f_cubic(x, m):
    return x[0]**3 + x[0] * x[1]**2 + 2 * x[1]


def f_exp(x, m):
    return m.exp(x[0] / 2) * x[1] + x[1]**3


def f_three(x, m):
    return x[0]**3 + x[0] * x[1]**2 + x[1] * x[2] + m.exp(x[2] / 3)


# Two inputs: E and central moments. (id, f, names, a, b, n)
MOMENT_CASES = [
    ("N-cubic-n2", f_cubic, ["N", "N"], [1.0, 2.0], [0.5, 0.3], 2),
    ("N-cubic-n3", f_cubic, ["N", "N"], [1.0, 2.0], [0.5, 0.3], 3),
    ("U-cubic-n2", f_cubic, ["U", "U"], [0.5, 1.5], [1.5, 2.5], 2),
    ("N-exp-n2", f_exp, ["N", "N"], [1.0, 2.0], [0.5, 0.3], 2),
    # Kumaraswamy shapes a, b on [0, 1]; the paper examples use K inputs
    ("K-exp-n2", f_exp, ["K", "K"], [2.0, 3.0], [2.5, 2.0], 2),
]
MOMENT_PARAMS = [("a", 0), ("b", 0), ("b", 1)]

# Three inputs, mixed distributions: Sobol' and Shapley.
SOBOL_CASE = ("NNU-three-n2", f_three, ["N", "N", "U"], [1.0, 2.0, 0.5], [0.5, 0.3, 1.5], 2)
SOBOL_PARAMS = [("a", 0), ("b", 0)]

# n = 3: the z1 z2^2 term of x1 x2^2 makes E[T | x1] depend on the spread of
# x2, so its sensitivity needs the e_1 * e_p cross term (missed until
# 2026-11-01). Every parameter of each input.
SOBOL_CASE_N3 = ("NUT-three-n3", f_three, ["N", "U", "T"], [1.0, 1.5, 0.2], [0.5, 2.5, 1.0], 3)
SOBOL_PARAMS_N3 = [("a", 0), ("b", 0), ("a", 1), ("b", 1), ("a", 2), ("b", 2)]

# Kumaraswamy inputs (n = 2: the sympy oracle is slow for K at n = 3).
SOBOL_CASE_K = ("KNK-three-n2", f_three, ["K", "N", "K"], [2.0, 1.0, 2.5], [2.5, 0.5, 3.0], 2)
SOBOL_PARAMS_K = [("a", 0), ("b", 2)]

SOBOL_CASES = [(SOBOL_CASE, SOBOL_PARAMS), (SOBOL_CASE_N3, SOBOL_PARAMS_N3),
               (SOBOL_CASE_K, SOBOL_PARAMS_K)]


def pid(param):
    return f"{param[0]}{param[1] + 1}"
