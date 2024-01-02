#date: 2024-01-02T16:52:44Z
#url: https://api.github.com/gists/4ef2072763da268b1b1953e102ad21f1
#owner: https://api.github.com/users/flying-sheep

from __future__ import annotations

import warnings

import pytest


doctest_marker = pytest.mark.usefixtures("suppress_env")


@pytest.fixture
def suppress_env() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)


def pytest_itemcollected(item: pytest.Item) -> None:
    if isinstance(item, pytest.DoctestItem):
        item.add_marker(doctest_marker)
