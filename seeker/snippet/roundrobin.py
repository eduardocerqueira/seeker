#date: 2021-10-26T17:15:44Z
#url: https://api.github.com/gists/1684e32474e8cb41090f96eea5077858
#owner: https://api.github.com/users/FartDraft

import collections
from typing import Generator, Iterable


def roundrobin(*iterables: Iterable) -> Generator:
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    iterators = collections.deque(map(iter, iterables))
    while iterators:
        try:
            while True:
                yield next(iterators[0])
                iterators.rotate(-1)
        except StopIteration:
            # Удалить "закончившийся" итератор
            iterators.popleft()