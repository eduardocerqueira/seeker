#date: 2023-06-08T17:01:37Z
#url: https://api.github.com/gists/28a4f7de4bda1b7e77da0cfa3eeb3a55
#owner: https://api.github.com/users/HariPrasad-1999

"""
This module demonstrates tests which are expected to pass. Run on the command
line and redirect the output to a text file with the following command:

pytest test_good.py > z_test_good_output.txt
"""

import pytest

def inc(x):
    """ This is the function we will test """
    if type(x) not in [int, float]:
        raise ValueError("input must be int or float")
    return x + 1

def test_inc():
    """ Test the function works the way we expect using an assert statement """
    assert inc(3) == 4

def test_error():
    """ Test the function raises the correct error using a "with pytest.raises"
    context manager """
    with pytest.raises(ValueError):
        inc("three")
