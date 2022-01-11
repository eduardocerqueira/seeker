#date: 2022-01-11T17:00:07Z
#url: https://api.github.com/gists/0db4130159e5e77be3b4250183e0bd28
#owner: https://api.github.com/users/andyroiiid

# https://www.fatalerrors.org/a/prd-critical-hit-algorithm-of-the-common-pseudo-random-algorithm-in-the-game.html

from math import ceil, isclose


def p_from_c(c: float) -> float:
    d_pre_success_p: float = 0
    d_p_e: float = 0
    n_max_fail: int = ceil(1 / c)
    for i in range(1, n_max_fail + 1):
        d_cur_p: float = min(1.0, i * c) * (1 - d_pre_success_p)
        d_pre_success_p += d_cur_p
        d_p_e += i * d_cur_p
    return 1 / d_p_e


def c_from_p(p: float) -> float:
    d_up: float = p
    d_low: float = 0
    d_p_last: float = 1
    while True:
        d_mid: float = (d_up + d_low) / 2
        d_p_tested: float = p_from_c(d_mid)

        if isclose(d_p_tested - d_p_last, 0):
            break

        if d_p_tested > p:
            d_up = d_mid
        else:
            d_low = d_mid

        d_p_last = d_p_tested
    return d_mid


granular: int = 20

for i in range(granular // 10, granular + 1):
    p: float = i / granular
    c: float = c_from_p(p)
    ps: [float] = []
    n: int = 1
    while True:
        adjusted_p = c * n
        ps.append(adjusted_p)
        if adjusted_p >= 1:
            break
        n += 1
    print(f"P={p:.2%}", ", ".join(f"P({i + 1})={p:.2%}" for i, p in enumerate(ps)))
