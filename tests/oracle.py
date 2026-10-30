"""Exact sympy oracle for the quantities of a Taylor series expansion.

Target (C), the default (decided with Sam 2026-10-30): every quantity of the
order-n expansion built once at the nominal mean m(p0), as an exact function
of one distribution parameter p that moves only the input distribution. This
is the derivative of the moments of the current surrogate, needing no extra
derivatives of f. center="moving" gives target (A) instead: the expansion is
rebuilt at m(p), which needs order n + 1 of f. Writing z = x - m(p),

    T_n(z; p) = sum_{|alpha| <= n} c_alpha(p) z^alpha,
    c_alpha(p) = d^alpha f(m(p)) / alpha!,

and every quantity below is a polynomial moment of T_n in z, which needs only
the central moments mu_{i,k}(p) of each input.
"""

from itertools import combinations, product
from math import factorial

import sympy as sp

P = sp.Symbol("p")


def _rv_mean_and_central_moments(name, a, b, kmax):
    """Mean and central moments 1..kmax of one input, as sympy expressions.

    Parameter conventions follow tseuqlib.rv_moments:
      'N': a = mean, b = standard deviation
      'U': support [a, b]
      'K': Kumaraswamy shape a and b on [0, 1]
    """
    if name == "N":
        mean = a
        mu = [sp.Integer(0) if k % 2 else b**k * sp.factorial2(k - 1) for k in range(1, kmax + 1)]
    elif name == "U":
        mean = (a + b) / 2
        mu = [sp.Integer(0) if k % 2 else ((b - a) / 2)**k / (k + 1) for k in range(1, kmax + 1)]
    elif name == "K":
        raw = [sp.Integer(1)] + [b * sp.beta(1 + sp.Integer(j) / a, b) for j in range(1, kmax + 1)]
        mean = raw[1]
        mu = [sum(sp.binomial(k, j) * (-1)**j * raw[k - j] * mean**j for j in range(k + 1))
              for k in range(1, kmax + 1)]
    else:
        raise ValueError(f"no oracle for distribution {name!r}")
    return mean, mu


class TSEOracle:
    """Exact quantities of T_n for f, independent inputs, and one parameter p.

    Args:
        f: callable taking a list of sympy symbols and returning an expression.
        names: distribution letter of each input.
        a, b: nominal parameter values (use exact rationals where possible).
        n: order of the expansion.
        param: (which, i) with which in {"a", "b"}: the parameter replaced by
            the symbol p; or None for no parameter.
        kmax: highest moment order needed (n times the highest moment power).
        center: "moving" builds T_n at the p-dependent mean m(p) (target A);
            "fixed" builds it once at the nominal mean m(p0) and moves only
            the input distribution (target C: the derivative of the moments
            of the current surrogate, needing no extra derivatives of f).
    """

    def __init__(self, f, names, a, b, n, param=None, kmax=None, center="fixed"):
        self.d = len(names)
        self.n = n
        a = [sp.nsimplify(v) for v in a]
        b = [sp.nsimplify(v) for v in b]
        self.p0 = None
        if param is not None:
            which, i = param
            tgt = a if which == "a" else b
            self.p0 = tgt[i]
            tgt[i] = P
        kmax = kmax or 4 * n
        self.means, self.mu = [], []
        for i, name in enumerate(names):
            m, mu = _rv_mean_and_central_moments(name, a[i], b[i], kmax)
            self.means.append(m)
            self.mu.append([sp.Integer(1)] + mu)  # mu[i][k], with mu[i][0] = 1
        self.x = sp.symbols(f"x1:{self.d + 1}")
        self.z = sp.symbols(f"z1:{self.d + 1}")
        fx = f(list(self.x))
        if center == "moving":
            self.T = self._tse(fx, n, self.means)
        elif center == "fixed":
            m0 = [m.subs(P, self.p0) if self.p0 is not None else m for m in self.means]
            # T_n about m0, in x; then x = m(p) + z, so z keeps central moments mu(p)
            Tx = self._tse(fx, n, m0).subs({zi: xi - mi for zi, xi, mi in zip(self.z, self.x, m0)})
            self.T = sp.expand(Tx.subs({xi: mi + zi for xi, mi, zi in zip(self.x, self.means, self.z)}))
        else:
            raise ValueError(f"center must be 'moving' or 'fixed', not {center!r}")

    def _tse(self, fx, n, center):
        at_mean = dict(zip(self.x, center))
        T = sp.Integer(0)
        for alpha in product(range(n + 1), repeat=self.d):
            if sum(alpha) > n:
                continue
            der = fx
            for xi, ai in zip(self.x, alpha):
                if ai:
                    der = sp.diff(der, xi, ai)
            coeff = der.subs(at_mean) / sp.prod([factorial(ai) for ai in alpha])
            T += coeff * sp.prod([zi**ai for zi, ai in zip(self.z, alpha)])
        return T

    # ---- expectation operators ---------------------------------------------
    def _expect(self, expr, keep=()):
        """E[expr | z_keep]: integrate out every z_i with i not in keep."""
        poly = sp.Poly(sp.expand(expr), *self.z)
        out = sp.Integer(0)
        for powers, c in poly.terms():
            term = c
            for i, k in enumerate(powers):
                if i in keep:
                    term *= self.z[i]**k
                else:
                    term *= self.mu[i][k]
            out += term
        return out

    def expectation(self):
        return self._expect(self.T)

    def central_moment(self, k):
        ET = self.expectation()
        return self._expect((self.T - ET)**k)

    def closed_variance(self, u):
        """Var(E[T | z_u]) for a tuple of input indices u."""
        if len(u) == 0:
            return sp.Integer(0)
        g = self._expect(self.T, keep=set(u))
        Eg = self._expect(g)
        return self._expect((g - Eg)**2)

    def partial_variances(self):
        """V_u of the ANOVA decomposition for every non-empty u, by Moebius
        inversion of the closed variances: V_u = sum_{v subset u} (-1)^{|u|-|v|} Vc_v."""
        Vc = {(): sp.Integer(0)}
        for r in range(1, self.d + 1):
            for u in combinations(range(self.d), r):
                Vc[u] = self.closed_variance(u)
        V = {}
        for r in range(1, self.d + 1):
            for u in combinations(range(self.d), r):
                V[u] = sum((-1)**(len(u) - len(v)) * Vc[v]
                           for s in range(len(u) + 1) for v in combinations(u, s))
        return V, Vc

    def sobol_indices(self):
        V, _ = self.partial_variances()
        Vt = self.central_moment(2)
        return {u: Vu / Vt for u, Vu in V.items()}

    def total_sobol_indices(self):
        Vt = self.central_moment(2)
        out = []
        for i in range(self.d):
            rest = tuple(j for j in range(self.d) if j != i)
            out.append(1 - self.closed_variance(rest) / Vt)
        return out

    def shapley_effects(self):
        """phi_i = sum over u not containing i of
        |u|! (d - |u| - 1)! / d! * (Vc(u + i) - Vc(u)) / V."""
        _, Vc = self.partial_variances()
        Vt = self.central_moment(2)
        d = self.d
        out = []
        for i in range(d):
            others = [j for j in range(d) if j != i]
            s = sp.Integer(0)
            for r in range(len(others) + 1):
                w = sp.Rational(factorial(r) * factorial(d - r - 1), factorial(d))
                for u in combinations(others, r):
                    ui = tuple(sorted(u + (i,)))
                    s += w * (Vc[ui] - Vc[u])
            out.append(s / Vt)
        return out

    # ---- evaluation --------------------------------------------------------
    def value(self, expr):
        e = expr.subs(P, self.p0) if self.p0 is not None else expr
        return float(sp.N(e, 30))

    def deriv(self, expr):
        return float(sp.N(sp.diff(expr, P).subs(P, self.p0), 30))
