"""Smoke-run the example scripts, as a user would, in a temporary copy (so
their plots and output files stay out of the repo), with empty input for
their input() pauses.

OUTCOMES lists what each example may do: None means it runs; a string is a
recorded error, for examples broken outside the library, and shows as
xfailed. Some depend on the stack (old: Python 3.9, numpy 1.21, matplotlib
3.7, a local pyoti build; CI: Python 3.13, numpy 2, matplotlib 3.10, pyoti
1.2.1). Any other outcome fails the test: a fix that makes an example run,
or a new error. Update the table then."""

import os
import shutil
import subprocess
import sys

import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS)

# Measured 2026-11-01 on both stacks. poly20_paper and sos, which are not in
# the paper and do not run, moved to examples/devel (not run here).
OUTCOMES = {
    "poly3_paper": [None],
    "sensitivity_check": [None],
    # computes every result; matplotlib 3.7 cannot parse a plot label, 3.10 can
    "ishigami_paper": [None, "ParseSyntaxException"],
    # runs with pyoti 1.2.1; the local pyoti build has no oti.abs
    "gfunction": [None, "module 'pyoti.sparse' has no attribute 'abs'"],
}

# Command-line arguments: gfunction's input count (it sweeps 1 to 100 by
# default, which needs the performance work).
ARGS = {"gfunction": ["3"]}


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    # the examples import tseuqlib with sys.path.append('../')
    root = tmp_path_factory.mktemp("examples_run")
    shutil.copytree(os.path.join(ROOT, "tseuqlib"), root / "tseuqlib",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(os.path.join(ROOT, "examples"), root / "examples",
                    ignore=shutil.ignore_patterns("__pycache__", "devel"))
    return root / "examples"


@pytest.mark.parametrize("name", OUTCOMES)
def test_example_runs(sandbox, name):
    env = dict(os.environ, MPLBACKEND="Agg")
    run = subprocess.run([sys.executable, "-B", f"{name}.py"] + ARGS.get(name, []), cwd=sandbox, env=env,
                         input="\n" * 1000, capture_output=True, text=True, timeout=900)
    tail = (run.stdout[-500:] + run.stderr[-1500:]).strip()
    allowed = OUTCOMES[name]
    if run.returncode == 0:
        assert None in allowed, f"{name} now runs: update OUTCOMES"
        return
    known = [e for e in allowed if e is not None and e in run.stderr]
    assert known, f"{name} fails differently than recorded:\n{tail}"
    pytest.xfail(f"known broken: {known[0]}")
