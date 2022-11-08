#date: 2022-11-08T17:18:47Z
#url: https://api.github.com/gists/5d5fdc235672b42d49e6907634064dfa
#owner: https://api.github.com/users/leaver2000

from app.core import py_func
from app._api import cy_func



def test_cy():
    assert cy_func(1) == 1
    
def test_py():
    assert py_func(1) == 1
    assert py_func(2) == 10