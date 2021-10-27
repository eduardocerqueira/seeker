#date: 2021-10-27T17:02:10Z
#url: https://api.github.com/gists/bf856061904d29dae1d57c17a8ded8d4
#owner: https://api.github.com/users/dougb

#!/usr/bin/env python3
# Base on code from this SO answer https://stackoverflow.com/a/17945009
from timeit import timeit
from random import shuffle

num = 10000
# r = 7

r = 10

a = [x for x in range(r)]
shuffle(a)


def in_test(iterable):
     for i in a:
         if i in iterable:
             pass


st = timeit(
    "in_test(iterable)",
    setup=f"from __main__ import in_test; iterable = set(range({r}))",
    number=num
)
print(f"Set:{st:2.6f} secs range:{r} iterations:{num}")

lt = timeit(
   "in_test(iterable)",
    setup=f"from __main__ import in_test; iterable = list(range({r}))",
    number=num
)
print(f"list:{lt:2.6f} secs range:{r} iterations:{num} set is {(lt/st):2.4f} faster.")



tt = timeit(
    "in_test(iterable)",
    setup=f"from __main__ import in_test; iterable = tuple(range({r}))",
    number=num
)
print(f"tuple:{tt:2.6f} secs range:{r} iterations:{num} set is {(tt/st):2.4f} faster.")