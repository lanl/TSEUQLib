"""Test configuration."""


def pytest_configure(config):
    # sympy's mpmath calls on the modern stack (mpmath 1.4+) emit this
    # hundreds of thousands of times; it is not about this code.
    config.addinivalue_line("filterwarnings", "ignore:bitcount function is deprecated:DeprecationWarning")
