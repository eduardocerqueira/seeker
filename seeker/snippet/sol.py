#date: 2023-02-06T17:08:08Z
#url: https://api.github.com/gists/dfd38f8211962c36e1424e3c17940972
#owner: https://api.github.com/users/benediktwerner

#!/usr/bin/env python3

# solver for the ? = 10 game where you get 4 numbers and the operations +, -, *, /, and optionally one pair of parentheses and need to get 10

import itertools, collections

count = collections.defaultdict(int)


def solve(vals, parens=True):
    sols = set()
    for a, b, c, d in map(lambda x: map(str, x), itertools.permutations(vals)):
        for x, y, z in itertools.product("+-*/", repeat=3):
            expr = a + x + b + y + c + z + d
            try:
                if eval(expr) == 10:
                    sols.add(expr)
            except ZeroDivisionError:
                pass
            if not parens:
                continue
            for i in range(3):
                for j in range(i, 3):
                    expr2 = (
                        expr[: 2 * i]
                        + "("
                        + expr[2 * i : 2 * (j + 1) + 1]
                        + ")"
                        + expr[2 * (j + 1) + 1 :]
                    )
                    try:
                        if eval(expr2) == 10:
                            sols.add(expr2)
                    except ZeroDivisionError:
                        pass
    return sols


for i, vals in enumerate(itertools.combinations_with_replacement(range(10), 4)):
    print(i)
    count[len(solve(vals))] += 1

print(sum(count.values()))

for k in sorted(count.keys()):
    print(k, count[k])
