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
]
MOMENT_PARAMS = [("a", 0), ("b", 0), ("b", 1)]

# Three inputs, mixed distributions: Sobol' and Shapley.
SOBOL_CASE = ("NNU-three-n2", f_three, ["N", "N", "U"], [1.0, 2.0, 0.5], [0.5, 0.3, 1.5], 2)
SOBOL_PARAMS = [("a", 0), ("b", 0)]


def pid(param):
    return f"{param[0]}{param[1] + 1}"
