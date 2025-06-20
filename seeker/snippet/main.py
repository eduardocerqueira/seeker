#date: 2025-06-20T17:14:30Z
#url: https://api.github.com/gists/592cb4b96d086e11ff8e2c22f05b3be1
#owner: https://api.github.com/users/mypy-play

from typing import Mapping, Sequence

class Foo:
    def __init__(self):
        # Very important, it should always contain three elements!
        self._my_list = [1,2,3]
        
    def get_list(self):
        return self._my_list
        

f = Foo()
l = f.get_list()
l.append(4)