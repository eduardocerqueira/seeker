#date: 2025-01-24T16:53:12Z
#url: https://api.github.com/gists/1c5a9b8bc7bb93ca1cb2aee37268f9c6
#owner: https://api.github.com/users/marcusadair

import os
import typing

import polars as pl
import polars.selectors as cs
import pytest

# these markers are enabled when `--with-{marker}` or PYTEST_WITH_{MARKER} != 0
CLI_MARKERS = ("slow", "very_slow")


def pytest_addoption(parser: pytest.Parser) -> None:
    """pytest hook function — adds custom cli options."""
    for mark in CLI_MARKERS:
        parser.addoption(f"--with-{mark}", f"Include {mark} tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Function]) -> None:
    """pytest hook function — modify collected test cases."""

    def chk_cli_markers(it: pytest.Function):
        for marker in CLI_MARKERS:
            if marker in it.keywords and not (
                config.getoption(f"--with-{marker.replace('_', '-')}")
                or os.getenv(f"PYTEST_WITH_{marker.upper()}", "0") != "0"
            ):
                it.add_marker(pytest.mark.skip(reason=f"Enabled by --with-{marker}"))
                return

    for item in items:
        chk_cli_markers(item)


@pytest.fixture(autouse=True)
def add_to_doctest_namespace(doctest_namespace: dict[str, typing.Any]) -> None:
    """Pre-import frequently used packages and dataset modules into doctest namespace."""
    doctest_namespace.update(dict(np=np, pl=pl, cs=cs))
