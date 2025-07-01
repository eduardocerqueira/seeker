#date: 2025-07-01T16:53:42Z
#url: https://api.github.com/gists/3b4c1e774be0cd1d5aafea410f228cb1
#owner: https://api.github.com/users/geky

#
# IPython:
# %run -n putil.py
#

import itertools as it
import functools as ft
import math as mt
import re


def p(old, new): return '%+.1f%%' % (100*((new-old)/old))

def pt(a, b, *,
        header=('code', 'stack', 'ctx'),
        sider=('', 'before:', 'after:')):
    rows = [[s] for s in sider]
    for i, (a_, b_) in enumerate(zip(a, b)):
        rows[0].append(header[i] if i < len(header) else '')
        rows[1].append('%s' % a_)
        rows[2].append('%s' % b_)
        rows[0].append('')
        rows[1].append('')
        rows[2].append('(%s)' % p(a_, b_))

    ws = [0 for c in rows[0]]
    for r in rows:
        for i, c in enumerate(r):
            ws[i] = max(ws[i], len(c))

    for r in rows:
        for i, (c, w) in enumerate(zip(r, ws)):
            if i == 0:
                print('  ', end='')
            elif i == 1 or (i % 2) == 0:
                print(' ', end='')
            else:
                print('  ', end='')
            if i == 0:
                print('%-*s' % (w, c), end='')
            else:
                print('%*s' % (w, c), end='')
        print()