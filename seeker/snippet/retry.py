#date: 2022-10-20T17:11:21Z
#url: https://api.github.com/gists/bb804dc003c87cd2348a4ea64220fbef
#owner: https://api.github.com/users/dutc

from functools import wraps
from itertools import islice, tee, zip_longest, chain, product
from collections import deque
from pandas import DataFrame

nwise = lambda g, *, n=2: zip(*(islice(g, i, None) for i, g in enumerate(tee(g, n))))
nwise_longest = lambda g, *, n=2, fv=object(): zip_longest(*(islice(g, i, None) for i, g in enumerate(tee(g, n))), fillvalue=fv)
first = lambda g, *, n=1: zip(chain(repeat(True, n), repeat(False)), g)
last = lambda g, *, m=1, s=object(): ((y[-1] is s, x) for x, *y in nwise_longest(g, n=m+1, fv=s))

def retry(*exc_types, tries=2): # XXX: a legitimately bad idea
    ''' allows you to dynamically retry a function and ignore certain exceptions '''
    exc_types = exc_types if exc_types else (Exception,)
    def dec(f):
        @wraps(f)
        def inner(*args, **kwargs):
            for is_last, _ in last(range(tries)):
                try:
                    return f(*args, **kwargs)
                except exc_types as e:
                    if is_last: raise
                    continue
        return inner
    return dec

@retry()
def f(AssertionError, xs):
    assert xs.popleft()

def res_or_err(f, *args, **kwargs):
    try: return f(*args, **kwargs)
    except Exception as e: return e

results = DataFrame.from_records([
    (try1, try2, not isinstance(res_or_err(f, deque([try1, try2])), Exception))
    for try1, try2 in product([True, False], repeat=2)
], columns='try1 try2 succeeds'.split()).set_index(['try1', 'try2']).squeeze(axis='columns')

print(results)