#date: 2025-11-17T17:09:55Z
#url: https://api.github.com/gists/e32b97140c9abcd074f0db0bbe930652
#owner: https://api.github.com/users/mypy-play

def foo1() -> list[tuple[int, int]]:
    l = []
    l.append((0, 1))
    return l
    
def foo2() -> list[int, int]
    l = []
    l.append([0, 1])
    return l