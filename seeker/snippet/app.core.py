#date: 2022-11-08T17:18:47Z
#url: https://api.github.com/gists/5d5fdc235672b42d49e6907634064dfa
#owner: https://api.github.com/users/leaver2000

def func_1():
    return 1


def func_any():
    return 10


def py_func(num: int) -> int:
    if num == 1:
        return func_1()
    else:
        return func_any()
