#date: 2022-03-28T16:49:21Z
#url: https://api.github.com/gists/12ebb105155227af55d51db9af006a4f
#owner: https://api.github.com/users/zohassadar

import typing
import itertools
import operator
import re



def expand_numerical_range(rngstr: str) -> list[int]:
    result = set()
    filtered = re.sub(r"[^\d,-]", "", rngstr)
    for group in filtered.split(","):
        if not (span := sorted([int(i) for i in group.split("-") if i])):
            continue
        begin, end = span[0], span[-1]
        result.update(set(range(begin, end + 1)))
    return sorted(list(result))


def collapse_numerican_range(numbers: typing.Iterable[int]) -> str:
    diff = lambda x: x[0] - x[1]
    indexed = enumerate(numbers)
    grouped = ([g[1] for g in group[1]] for group in itertools.groupby(indexed, diff))
    pre_results = []
    for group in grouped:
        if len(group) == 1:
            pre_results.append(str(group[0]))
            continue
        converted = map(str, (group[0], group[-1]))
        pre_results.append("-".join(converted))
    return ",".join(pre_results)
