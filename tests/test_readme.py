"""The README's Python code runs as written: its blocks, in order, in one
namespace, the way a reader would paste them."""

import contextlib
import io
import os
import re

TESTS = os.path.dirname(os.path.abspath(__file__))


def test_readme_code_runs():
    with open(os.path.join(TESTS, "..", "README.md")) as fh:
        blocks = re.findall(r"```python\n(.*?)```", fh.read(), flags=re.S)
    assert blocks, "the README has no Python blocks"
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()) as out:
        for code in blocks:
            exec(compile(code, "README.md", "exec"), ns)
    # the sensitivity block ends with the variance and its derivative
    var, dvar = (float(v) for v in out.getvalue().split()[-2:])
    assert var > 0 and dvar > 0  # more spread in the load, more spread in the deflection
