#date: 2022-10-18T17:17:36Z
#url: https://api.github.com/gists/fe265659824993d49c2460187b86417f
#owner: https://api.github.com/users/eriknw

import sys
import networkx as nx
from collections import defaultdict
from types import FunctionType


def fullname(func):
    return f"{func.__module__}.{func.__name__}"


def is_nxfunc(func):
    return (
        callable(func)
        and isinstance(func, FunctionType)
        and func.__module__.startswith("networkx.")
        and not func.__name__.startswith("_")
    )


info = {}
other = set()
for modname, module in sys.modules.items():
    cur = set()
    if (
        not modname.startswith("networkx.")
        and modname != "networkx"
        or "tests" in modname
    ):
        continue
    for key, val in vars(module).items():
        if key.startswith("_") or not callable(val) or isinstance(val, type):
            continue
        if is_nxfunc(val):
            cur.add(fullname(val))
        elif callable(val):
            try:
                other.add(fullname(val))
            except Exception:
                print("ERROR:", key, val)
    if cur:
        info[modname] = cur

print("Total number of functions:", len(set().union(*info.values())))
print(
    "Number of functions in `networkx.algorithms`:",
    len(
        [x for x in set().union(*info.values()) if x.startswith("networkx.algorithms")]
    ),
)
print()
for key in sorted(info):
    print(key)
    for k in sorted(info[key]):
        print("   ", k)

print()
print("Duplicate names:")
d = defaultdict(set)
for x in set().union(*info.values()):
    d[x.split(".")[-1]].add(x)
d = {k: v for k, v in d.items() if len(v) > 1}
for k in sorted(d):
    print(k)
    for kk in sorted(d[k]):
        print("   ", kk)
