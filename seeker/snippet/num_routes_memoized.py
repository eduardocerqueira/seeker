#date: 2022-06-08T17:04:28Z
#url: https://api.github.com/gists/420e08e4bce815c7b75cbb653a0bbfbe
#owner: https://api.github.com/users/noshiro-pf

import argparse


def combination(n: int, k: int) -> int:
    ret: int = 1
    for i in range(0, k):
        ret *= n - i
    for i in range(0, k):
        ret //= i + 1
    return ret


def catalan_number(n: int) -> int:
    return combination(2 * n, n) // (n + 1)


def P(n: int, k: int) -> int:
    if k <= 1:
        return 1
    if memo[n][k] is not None:
        return memo[n][k]

    if k >= n:
        memo[n][n] = catalan_number(n)
        return memo[n][n]

    ret: int = 0
    for i in range(0, n - 1):
        ret += P(i, k) * P(n - i - 1, k)
    ret += P(n - 1, k - 1)
    memo[n][k] = ret
    return ret


parser = argparse.ArgumentParser()

parser.add_argument("n", type=int)
parser.add_argument("k", type=int)

args = parser.parse_args()


memo = [[None for _i in range(0, args.k + 1)] for _j in range(0, args.n + 1)]

print(P(args.n, args.k))
