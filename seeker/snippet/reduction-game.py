#date: 2022-11-01T17:01:27Z
#url: https://api.github.com/gists/7d69dfb869466f308f6bcf7d91991afb
#owner: https://api.github.com/users/anandology

from functools import reduce

LEN = lambda items: reduce(lambda n, _: n + 1, items, 0)

REVERSE = lambda items: reduce(lambda rev, item: [item, *rev], items, [])

MAP = lambda func, items: reduce(lambda result, item: [*result, func(item)], items, [])
FILTER = lambda func, items: reduce(lambda result, item: [*result, item] if func(item) else result, items, [])

APPEND = lambda items, value: reduce(lambda result, item: [*result, item], [value], items)
EXTEND = lambda items, values: reduce(lambda result, item: [*result, item], values, items)

APPEND = lambda items, value: [*items, value]
EXTEND = lambda items, values: [*items, *values]

COUNT = lambda items, value: reduce(lambda result, item: result+int(item==value), items, 0)
REMOVE = lambda items, value: reduce(lambda result, item: [*result] if item == value else [*result, item], items, [])

CAR = lambda a, *_: a
CDR = lambda _, *a: a
GETITEM = lambda items, index: CAR(*reduce(lambda result, _: CDR(*result), [0]*index, items))

# GETITEM with CAR and CDR inlined
GETITEM = lambda items, index: (lambda a, *_: a)(*reduce(lambda result, _: (lambda _, *a: a)(*result), [0]*index, items))

INSERT = lambda items, item: \
    [*reduce(lambda result, value:  [*result, value] if value < item else result, items, []),
    item,
    *reduce(lambda result, value:  [*result, value] if value >= item else result, items, [])]

SORT = lambda items: reduce(lambda result, item: INSERT(result, item), items, [])

# SORT without INSERT inlined
SORT = \
  lambda items: reduce(
    lambda result, item: \
      (lambda items, item: \
        [*reduce(lambda result, value:  [*result, value] if value < item else result, items, []),
        item,
        *reduce(lambda result, value:  [*result, value] if value >= item else result, items, [])])(result, item),
      items, [])

def t(func_name, *args):
    func = globals()[func_name]
    result = func(*args)
    print(f">>> {func_name}{args}")
    print(result)
    print()

items = [1, 2, 3, 4]
t("LEN", items)
t("REVERSE", items)
t("MAP", lambda x: x*x, items)
t("FILTER", lambda x: x%2==0, items)
t("APPEND", items, 5)
t("EXTEND", items, [5, 6, 7])
t("COUNT", items, 3)
t("REMOVE", items, 3)
t("GETITEM", items, 2)
t("INSERT", items, 2.5)
t("SORT", [1,8,2,3,-5])