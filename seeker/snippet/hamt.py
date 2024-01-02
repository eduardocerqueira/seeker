#date: 2024-01-02T16:55:03Z
#url: https://api.github.com/gists/28d6260828dc734d44de54731831d952
#owner: https://api.github.com/users/mbillingr

from __future__ import annotations
import ctypes
import dataclasses
from typing import Any

LEAF_SIZE = 32
LEAF_MASK = LEAF_SIZE - 1
HASH_BITS = 64


class Trie:
    pass


@dataclasses.dataclass
class Leaf(Trie):
    key: Any
    val: Any


@dataclasses.dataclass
class Node(Trie):
    mapping: int
    subtrie: list[Trie]


def empty() -> Node:
    return Node(0, [])


def lookup(key: Any, trie: Node) -> Any:
    k = uhash(key)
    return lookup_(key, k, trie)


def lookup_(key: Any, k: int, trie: Node) -> Any:
    idx = k & LEAF_MASK
    mask_bit = 1 << idx
    if not trie.mapping & mask_bit:
        raise KeyError(key)
    idx_ = ctpop(trie.mapping & (mask_bit - 1))
    match trie.subtrie[idx_]:
        case Leaf(key_, val_):
            if key_ == key:
                return val_
            else:
                raise KeyError(key)
        case Node() as child:
            return lookup_(key, k // LEAF_SIZE, child)


def insert(key: Any, value: Any, trie: Node) -> Node:
    k = uhash(key)
    return insert_(key, value, k, 1, trie)


def insert_(key: Any, value: Any, k: int, depth: int, trie: Node) -> Node:
    idx = k & LEAF_MASK
    mask_bit = 1 << idx
    idx_ = ctpop(trie.mapping & (mask_bit - 1))

    if trie.mapping & mask_bit:
        match trie.subtrie[idx_]:
            case Leaf(key_, _) if key_ == key:
                new_child = Leaf(key, value)
            case Leaf() as leaf:
                new_child = split_(
                    Leaf(key, value), k // LEAF_SIZE, leaf, uhash(leaf.key) // (depth * LEAF_SIZE)
                )
            case Node() as child:
                new_child = insert_(key, value, k // LEAF_SIZE, depth * LEAF_SIZE, child)
        new_subtrie = trie.subtrie.copy()
        new_subtrie[idx_] = new_child
        return Node(trie.mapping, new_subtrie)
    else:
        leaf = Leaf(key, value)
        new_mapping = trie.mapping | mask_bit
        new_subtrie = trie.subtrie.copy()
        new_subtrie.insert(idx_, leaf)
        return Node(new_mapping, new_subtrie)


def split_(leaf1: Leaf, k1: int, leaf2: Leaf, k2: int) -> Node:
    idx1 = k1 & LEAF_MASK
    idx2 = k2 & LEAF_MASK
    mb1 = 1 << idx1
    mb2 = 1 << idx2
    if idx1 == idx2:
        return Node(mb1, [split_(leaf1, k1 // LEAF_SIZE, leaf2, k2 // LEAF_SIZE)])
    if mb1 < mb2:
        subtree = [leaf1, leaf2]
    else:
        subtree = [leaf2, leaf1]
    return Node(mb1 | mb2, subtree)


def uhash(x: Any) -> int:
    """Make sure we get a unsigned hash value"""
    return ctypes.c_size_t(hash(x)).value


def ctpop(x: int) -> int:
    count = 0
    while x > 0:
        count += x & 1
        x //= 2
    return count


assert ctpop(0) == 0
assert ctpop(1) == 1
assert ctpop(2) == 1
assert ctpop(3) == 2
assert ctpop(4) == 1
assert ctpop(5) == 2
assert ctpop(255) == 8


def show_trie(trie: Trie, indent=""):
    match trie:
        case Leaf(k, v):
            print(indent, bin(uhash(k)), k, ":", v)
        case Node(m, s):
            print(indent, bin(m))
            for t in s:
                show_trie(t, indent + "  ")


m = empty()
for k in range(100):
    m = insert(str(k), k, m)
show_trie(m)

assert lookup("0", m) == 0
assert lookup("1", m) == 1
assert lookup("2", m) == 2
assert lookup("99", m) == 99


if False:
    import random
    import time
    from matplotlib import pyplot as plt

    m = empty()
    times = []
    for _ in range(10000):
        k = str(random.randint(0, 2**60))
        a = time.time()
        m = insert(str(k), k, m)
        b = time.time() - a
        times.append(b)
    for _ in range(10000):
        k = str(random.randint(0, 2**60))
        a = time.time()
        _ = insert(str(k), k, m)
        b = time.time() - a
        times.append(b)

    plt.plot(times)
    plt.show()
