#date: 2021-10-18T16:59:20Z
#url: https://api.github.com/gists/3554b7ffc0d61d60b38e44a12b62bc7d
#owner: https://api.github.com/users/jonashaag

import pytest


def mark_fixture(mark, *args, **kwargs):
    """Decorator to mark a fixture.

    Usage:
        @mark_fixture(pytest.mark.slow, scope="session", ...)
        def my_fixture():
            ...
        def test_xyz(my_fixture):
            # This test will be marked slow.
            ...
    """
    return pytest.fixture(*args, **kwargs, params=[pytest.param("dummy", marks=mark)])