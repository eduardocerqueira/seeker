#date: 2021-11-12T17:02:09Z
#url: https://api.github.com/gists/fedf3788e0738e4d5718e1e67ff83276
#owner: https://api.github.com/users/and-semakin

def no_check(a, b):
    return a / b


def try_check(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None


def if_check(a, b):
    if b != 0:
        return a / b
    else:
        return None


%timeit no_check(1, 1)
%timeit try_check(1, 1)
%timeit try_check(1, 0)
%timeit if_check(1, 1)
%timeit if_check(1, 0)
