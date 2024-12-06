#date: 2024-12-06T17:06:58Z
#url: https://api.github.com/gists/5a1e43d15e766b314ee6b258eceb62e1
#owner: https://api.github.com/users/RaphaelSiller

from scipy.special import binom


def sum_part(n: int, t: int):
    sum = 0
    for k in range(0, t):
        sum += binom(n, k)
    return sum


def calc_highest_hamming(n: int, w: int) -> int:
    t = 1
    while (2**n) / sum_part(n, t) >= w:
        t += 1
    #t -= 1
    #return 2 * t + 1  # t = (d - 1) / 2 <=> d = 2 * t + 1
    return 2 * t # t = (d - 1) / 2 <=> d = 2 * t + 1


if __name__ == "__main__":
    n = int(input("number of bits: "))
    w = int(input("number of required words: "))
    print(
        f"the highest possible hamming distance for a codebook with {w} words and a bitlength of {n} is {calc_highest_hamming(n, w)}"
    )
