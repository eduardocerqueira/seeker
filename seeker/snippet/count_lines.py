#date: 2023-06-05T16:57:16Z
#url: https://api.github.com/gists/3053ab5a486efc476869d5541da49ecc
#owner: https://api.github.com/users/ap--

import mmap

def count_lines(fn: str | Path) -> int:
    """count lines in file"""
    with open(fn, "rb") as f:
        buf = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        return sum(1 for _ in iter(buf.readline, b""))
