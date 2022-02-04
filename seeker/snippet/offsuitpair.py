#date: 2022-02-04T16:46:12Z
#url: https://api.github.com/gists/7a85fe085582456a2497c914e721aaa8
#owner: https://api.github.com/users/BowlOfRed

from itertools import product
import random
from collections import Counter

ranks = 'A23456789TJQK'
suits = 'DCHS'

def deck():
    return list(product(ranks, suits))

def pairwise(t):
    it = iter(t)
    return zip(it, it)

def offsuit_nonpair(c1, c2):
    return c1[0] != c2[0] and c1[1] != c2[1]

d = deck()
player_pairs = Counter()
for attempt in range(5000000):
    while True:
        random.shuffle(d)
        if offsuit_nonpair(d[0], d[1]):
            break

    hands = pairwise(d)
    next(hands) # discard my hand
    for player, hand in enumerate(hands):
        if hand[0][0] == hand[1][0]:
            player_pairs[player] += 1
            break

pairs_hit = 0
for player in range(10):
    pairs_hit += player_pairs[player]
    print(f"{player + 1} - {pairs_hit/attempt:.4f} - {1 - pairs_hit/attempt:.4f}")