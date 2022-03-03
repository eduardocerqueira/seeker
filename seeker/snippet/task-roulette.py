#date: 2022-03-03T17:02:50Z
#url: https://api.github.com/gists/80a4d3d5050c517beff73778c4778de0
#owner: https://api.github.com/users/Nircek

#!/usr/bin/env python3
import random

N = 100
bag, weights = [], []

class BreakException(Exception):
    """Break the line parsing loop"""
class ContinueException(Exception):
    """Continue the line parsing loop"""

def ignore(line):
    print(f'# ignoring `{line}`')
    raise ContinueException()


def parseLine(line):
    stripped = line.replace('-', '').strip()
    if stripped == 'END':
        raise BreakException()
    if stripped == '':
        raise ContinueException()
    l = line.split('x ', 1)
    if len(l) != 2:
        ignore(line)
    q, x = l[0].rsplit(' ', 1)[-1], l[1]
    try:
        q = int(q)
    except ValueError:
        ignore(line)
    if q <= 0:
        raise ContinueException()
    return q, x

print('''Paste your list like this:
```
 - 1x thing
```
and end it with the `END` keyword or send the EOF signal.''', end=3*'\n')
try:
    while True:
        try:
            w, b = parseLine(input())
            bag += [b]
            weights += [w]
        except ContinueException:
            pass
except (BreakException, EOFError):
    pass

assert bag
if sum(weights) <= N:
    bag = sum([w*[b] for w, b in zip(weights, bag)], []) + (random.choices(bag, weights, k=N-sum(weights)) if sum(weights) < N else [])
if len(bag) < N:
    bag += random.choices(bag, weights, k=N-len(bag))
if len(bag) > N:
    bag = random.choices(bag, weights, k=N)
bag.sort()

assert len(bag) == N 
siz = len(str(N-1))
for i, e in enumerate(bag):
    j = str(i).rjust(siz, '0')
    print(f'{j}. {e}')
