#date: 2023-12-11T17:05:43Z
#url: https://api.github.com/gists/474a31996c5c53ad0af26a863711902a
#owner: https://api.github.com/users/lmmx

import greenery
from greenery import Pattern, parse


def find_common_prefix(p1: Pattern, *patterns) -> tuple[Pattern, Pattern]:
    p1_concs = next(iter(p1.concs))
    common_prefixes = [p1_concs.common(next(iter(p.concs))) for p in patterns]
    total_common = [pfx for pfx in common_prefixes if pfx.mults]  # drop empty patterns
    assert len(total_common) < 2  # only handle simplest case for now: 0 or 1 prefix
    common_prefix = str(total_common[0]) if total_common else ""
    derivate = p1.derive(common_prefix)
    return common_prefix, derivate


def find_minimal_pattern(p1: Pattern, *patterns) -> str:
    common_prefix, derivate = find_common_prefix(p1, *patterns)
    dc = next(iter(derivate.concs))
    dcm = next(iter(dc.mults))
    multi_char = dcm.multiplicand
    char = next(multi_char.get_chars())
    minimal_pattern = parse(common_prefix + char)
    return str(minimal_pattern)


run_abc = True

A = parse("aab").reduce()  # shortest unique substring: 'aa'
B = parse("abc").reduce()  # shortest unique substring: 'ab'
C = parse("b..").reduce()  # shortest unique substring: 'b'

X = parse("xx").reduce()  # shortest unique substring: 'x'
Y = parse("yy").reduce()  # shortest unique substring: 'y'

if run_abc:
    assert (abc := find_minimal_pattern(A, B, C)) == "aa"
    assert (bac := find_minimal_pattern(B, A, C)) == "ab"
    assert (cab := find_minimal_pattern(C, A, B)) == "b"
    print(A, f"---> {abc!r}")
    print(B, f"---> {bac!r}")
    print(C, f"---> {cab!r}")

print()

assert (xy := find_minimal_pattern(X, Y)) == "x"
assert (yx := find_minimal_pattern(Y, X)) == "y"
print(X, f"---> {xy!r}")
print(Y, f"---> {yx!r}")
