#date: 2023-06-08T17:01:37Z
#url: https://api.github.com/gists/28a4f7de4bda1b7e77da0cfa3eeb3a55
#owner: https://api.github.com/users/HariPrasad-1999

"""
This module demonstrates tests which are expected to fail. Run on the command
line and redirect the output to a text file with the following command:

pytest test_bad.py > z_test_bad_output.txt
"""

import pytest

def inc(x):
    """ This is the function we will test """
    if type(x) not in [int, float]:
        raise ValueError("input must be int or float")
    return x + 1

def test_wrong_output():
    """ This test fails because the assert statement is False """
    assert inc(3) == 5
    
def test_wrong_input():
    """ This test fails because an uncaught error is raised """
    assert inc("three") == 4

def test_wrong_error():
    """ This test fails because the wrong error is caught (effecetively this is
    the same as the previous example: the test fails because an uncaught error
    is raised) """
    with pytest.raises(RuntimeError):
        inc("three")

def test_error_wrong_input():
    """ This test fails because it tries to catch an error which hasn't been
    raised """
    with pytest.raises(ValueError):
        inc(3)
