#date: 2024-09-09T17:08:38Z
#url: https://api.github.com/gists/7c1e8996114ece797c696a021ef75a13
#owner: https://api.github.com/users/docsallover

import pytest

def add(x, y):
    return x + y

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

pytest.main()