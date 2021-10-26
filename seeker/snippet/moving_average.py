#date: 2021-10-26T17:02:10Z
#url: https://api.github.com/gists/1bf874b5be527ff186ec60a2b6fc5811
#owner: https://api.github.com/users/FartDraft

import collections
import itertools
from typing import Generator, Iterable


def moving_average(iterable: Iterable, n: int=3) -> Generator:
    # moving_average([40, 30, 50, 46, 39, 44]) --> 40.0 42.0 45.0 43.0
    d = collections.deque(itertools.islice((it := iter(iterable)), n - 1))
    d.appendleft(0)
    s = sum(d)
    for elem in it:
        s += elem - d.popleft()
        d.append(elem)
        yield s / n