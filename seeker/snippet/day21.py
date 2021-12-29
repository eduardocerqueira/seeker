#date: 2021-12-29T16:59:39Z
#url: https://api.github.com/gists/5d7c64821111be5c7456b6f2cfc262a9
#owner: https://api.github.com/users/joshbduncan

import functools
import itertools


data = open("day21.in").read().strip().split("\n")

# part 1
p1, p2 = int(data[0][-1]), int(data[1][-1])
s1 = s2 = 0
die = rolls = 0
while True:
    die = die % 100 + 1
    if die % 2:  # player 1
        p1 += sum([die, die + 1, die + 2])
        s1 += p1 % 10 if p1 % 10 else 10
    else:  # player 2
        p2 += sum([die, die + 1, die + 2])
        s2 += p2 % 10 if p2 % 10 else 10
    die += 2
    rolls += 3
    if s1 >= 1000 or s2 >= 1000:
        break
print(f"Part 1: {min(s1, s2) * rolls}")


# part 2
@functools.lru_cache(maxsize=None)
def play_out(p1, s1, p2, s2):
    w1 = w2 = 0
    for m1, m2, m3 in itertools.product((1, 2, 3), (1, 2, 3), (1, 2, 3)):
        p1_copy = (p1 + m1 + m2 + m3) % 10 if (p1 + m1 + m2 + m3) % 10 else 10
        s1_copy = s1 + p1_copy
        if s1_copy >= 21:
            w1 += 1
        else:
            w2_copy, w1_copy = play_out(p2, s2, p1_copy, s1_copy)
            w1 += w1_copy
            w2 += w2_copy
    return w1, w2


p1, p2 = int(data[0][-1]), int(data[1][-1])
s1 = s2 = 0
print(f"Part 2: {max(play_out(p1, s1, p2, s2))}")