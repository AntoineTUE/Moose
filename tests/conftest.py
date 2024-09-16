import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--lmfit",
        action="store_true",
        dest="lmfit",
        default=False,
        help="enable tests for integration testing with lmfit as fitting backend",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "lmfit: enable tests with lmfit")
    if not config.option.lmfit:
        config.option.markexpr = "not lmfit"
