#date: 2023-06-20T16:29:42Z
#url: https://api.github.com/gists/0917a093e15b354432ca47559d1b3de8
#owner: https://api.github.com/users/mypy-play


from typing import Callable

class E:
    pass

ECB = Callable[[E], None]
NCB = Callable[[], None]

def t(cb: ECB | NCB):
    if isinstance(cv, ECB):
        cb(E())
    else:
        cb()

def ecb(e: E):
    pass
def ncb():
    pass

t(ecb)
t(ncb)
