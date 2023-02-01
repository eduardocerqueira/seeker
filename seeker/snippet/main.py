#date: 2023-02-01T16:45:57Z
#url: https://api.github.com/gists/f5e21caa675c153d4f8bda6b26e7451e
#owner: https://api.github.com/users/mypy-play

from typing import *

MyString = NewType("MyString", str)

def foo(bar: MyString):
    print(bar)
    
foo("blah")