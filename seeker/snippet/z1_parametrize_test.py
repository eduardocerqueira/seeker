#date: 2023-06-08T17:01:37Z
#url: https://api.github.com/gists/28a4f7de4bda1b7e77da0cfa3eeb3a55
#owner: https://api.github.com/users/HariPrasad-1999

import pytest

@pytest.mark.parametrize("single_arg", [2, 4, 6, 7])
def test_single_arg_even(single_arg):
    assert single_arg % 2 == 0

@pytest.mark.parametrize(
    "arg1, arg2, arg3",
    [(1, 2, 3), (4, 5, 9), (10, 11, 12)]
)
def test_multiple_args_sum(arg1, arg2, arg3):
    assert arg1 + arg2 == arg3

@pytest.mark.parametrize("x", [1, 2, 3, 4])
@pytest.mark.parametrize("y", [0, 2, 4, 10])
def test_stacked_parameters(x, y):
    assert x * x + y * y < 10 * 10
    