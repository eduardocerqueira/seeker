#date: 2021-11-16T17:03:58Z
#url: https://api.github.com/gists/fffcf5a7288584e2b6639f4317a22a90
#owner: https://api.github.com/users/stjordanis

"""
Reorder site-packages ahead of Extras and lib-dynload.
Two implementations:

1. puts site-packages ahead of stdlib (technically hazardous,
but not really an issue).
2. is more conservative, only demoting Extras below site-packages.

Add this to ~/Library/Python/2.7/lib/python/site-packages/sitecustomize.py
"""

import sys

mode = 'promote site-packages'

if mode == 'promote site-packages':
    # This puts the site-packages blob ahead of the stdlib
    # this is technically hazardous, because it allows you to override the stlib.
    # but you can do that anyway from the script path,
    # so it's not exactly a new problem.
    
    # find the first stdlib entry
    for sys_prefix_index, path in enumerate(sys.path):
        if path.startswith(sys.prefix):
            break
    
    # find everything not in the stdlib, and move it to higher priority.
    # this places site-packages ahead of stdlib, Extras, etc.
    for idx in range(sys_prefix_index, len(sys.path)):
        path = sys.path[idx]
        if not path.startswith(sys.prefix):
            sys.path.pop(idx)
            sys.path.insert(sys_prefix_index, path)
            sys_prefix_index += 1

elif mode == 'demote Extras':
    # more conservative approach.
    # this just demotes the Extras path below site-package,
    # allowing override of the shipped numpy, scipy, etc.
    # it does not address the readline issue.
    
    extras = sys.prefix + '/Extras'
    demoted = []
    idx = 0
    while idx < len(sys.path):
        path = sys.path[idx]
        if path.startswith(extras):
            demoted.append(sys.path.pop(idx))
        else:
            idx += 1
    sys.path.extend(demoted)
    