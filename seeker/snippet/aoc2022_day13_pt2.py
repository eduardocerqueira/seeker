#date: 2022-12-13T16:46:15Z
#url: https://api.github.com/gists/66ccfc1c7feae01e8b965920b130de35
#owner: https://api.github.com/users/dossett

import sys
from functools import cmp_to_key
from math import prod


def compare(left: list, right: list) -> int:

    # Special case this? Two empty lists should be considered the same
    if isinstance(left, list) and isinstance(right, list) and len(left) == 0 and len(right) == 0:
        return 0

    for i, l in enumerate(left):
        # if the right runs out when we still need it, it's not in correct order
        try:
            r = right[i]
        except IndexError:
            return 1
        if not isinstance(l, list) and not isinstance(r, list):
            if not l == r:
                return l - r
        else:
            val = compare(l if isinstance(l, list) else [l],
                           r if isinstance(r, list) else [r])
            if val == 0:
                continue
            else:
                return val

    return -1


def solve_it():
    all_packets = []
    for line in sys.stdin:
        line = line.strip()
        if len(line) > 0:
            all_packets.append(eval(line))

    dividers = [[[2]], [[6]]]
    sorted_packets = sorted(all_packets + dividers, key=cmp_to_key(compare))
    for p in sorted_packets:
        print(p)

    print(prod([sorted_packets.index(d) + 1 for d in dividers]))

solve_it()
