"""pyoti's basis limit. The library's copy of pyoti's capability table must
match the installed pyoti, and a quantity past the limit must raise a
ValueError instead of crashing pyoti. The crash cases run in a subprocess,
so a regression fails one test instead of killing the session."""

import os
import subprocess
import sys
import textwrap

import pytest

from library_adapter import uoti

TESTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS)


def _installed_table():
    # pyoti prints the table from C through a buffered stdout: read it from a
    # separate process
    out = subprocess.run([sys.executable, "-B", "-c",
                          "import pyoti.core as c; c.get_dHelp().print_capabilities()"],
                         capture_output=True, text=True, check=True).stdout
    rows = [line.split() for line in out.splitlines()]
    return {int(r[0]): int(r[1]) for r in rows if len(r) == 3 and r[0].isdigit()}


def test_table_matches_installed_pyoti():
    table = _installed_table()
    assert max(table) == uoti.PYOTI_MAX_ORDER
    for order in range(1, uoti.PYOTI_MAX_ORDER + 1):
        assert uoti.pyoti_max_bases(order) == table[order], f"order {order}"
    assert uoti.pyoti_max_bases(uoti.PYOTI_MAX_ORDER + 1) == 0


# mu4 of an order-3 expansion needs truncation order 12, where pyoti allows 5
# bases. Standard-deviation parameters enter through the moments only, so the
# first case checks that their bases are counted too. Six inputs already fail
# when the joint moments are built, at order 11, the first order allowing 5.
PAST_LIMIT = {
    "3-inputs-3-sd-params": ("['N'] * 3", "[('b', 0), ('b', 1), ('b', 2)]",
                             "at most 5 OTI bases at truncation order 12, and this needs 6"),
    "6-inputs-no-params": ("['N'] * 6", "None",
                           "at most 5 OTI bases at truncation order 11, and the joint moments of 6 inputs need 6"),
}

SCRIPT = """
import sys
sys.path[:0] = [{root!r}, {tests!r}]
from library_adapter import build

def f(x, m):
    return sum(xi**4 for xi in x) + x[0] * x[-1] * x[1]

names = {names}
d = len(names)
try:
    tq, y, _ = build(f, names, [1.0] * d, [0.5] * d, 3, param={params})
    tq.central_moment(y, 4)
    print("no error")
except ValueError as exc:
    print("ValueError:", exc)
"""


@pytest.mark.parametrize("case", PAST_LIMIT, ids=list(PAST_LIMIT))
def test_past_limit_raises_instead_of_crashing(case):
    names, params, message = PAST_LIMIT[case]
    code = textwrap.dedent(SCRIPT.format(root=ROOT, tests=TESTS, names=names, params=params))
    run = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, f"crashed ({run.returncode}): {run.stdout[-300:]} {run.stderr[-300:]}"
    assert "ValueError: pyoti supports " + message in run.stdout, run.stdout
