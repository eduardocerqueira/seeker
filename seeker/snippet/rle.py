#date: 2022-07-04T02:37:14Z
#url: https://api.github.com/gists/624747fec69783b156799dc1b1cf6bbc
#owner: https://api.github.com/users/duyixian1234

from itertools import takewhile
from typing import TypeVar

T = TypeVar("T", str, bytes)


def parse_raw(raw: T) -> tuple[tuple[T, int], T]:
    first, rest = raw[0], raw[1:]
    index = 0
    while index < len(rest) and rest[index] == first:
        index += 1
    return (first, index + 1), rest[index:]


def compress(raw: T) -> T:
    result = []
    while raw:
        (ch, count), raw = parse_raw(raw)
        result.append((str(count) if count else "") + ch)
    return "".join(result)


def parse_compressed(compressed: T) -> tuple[tuple[T, int], T]:
    count_part = "".join(takewhile(lambda x: x.isdigit(), compressed))
    count = int(count_part) if count_part else 1
    ch = compressed[len(count_part)]
    return (ch, count), compressed[len(count_part) + 1 :]


def decompress(raw: T) -> T:
    result = []
    while raw:
        (count, ch), raw = parse_compressed(raw)
        result.append(ch * count)
    return "".join(result)


def test_rle():
    assert compress("aabbccddeeff") == "2a2b2c2d2e2f"
    assert decompress("3ab4c") == "aaabcccc"
