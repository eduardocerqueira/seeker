#date: 2023-01-25T17:02:45Z
#url: https://api.github.com/gists/5f316441a3562d99b5805ebd7f29f4fb
#owner: https://api.github.com/users/yoniLavi

from functools import reduce
from operator import and_

lines = [
    "vJrwpWtwJgWrhcsFMMfFFhFp",
    "jqHRNqRjqzjGDLGLrsFMfFZSrLrFZsSL",
    "PmmdzqPrVvPwwTWBwg",
    "wMqvLMZHhHMvwLHjbvcjnnSBnvTQFn",
    "ttgJtRGJQctTZtZT",
    "CrZsJsPPZsGzwwsLwLmpwMDw",
]

priority = lambda c: ord(c) - (96 if c.islower() else 38)
triples = zip(lines[::3], lines[1::3], lines[2::3])
common_items = (reduce(and_, map(set, t)).pop() for t in triples)
print(sum(map(priority, common_items)))