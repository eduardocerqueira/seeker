#date: 2023-06-13T16:55:34Z
#url: https://api.github.com/gists/1bf9bb1848585659ee7a7e95f0de6d47
#owner: https://api.github.com/users/panstx

#!/usr/bin/env python3

# BSD Zero Clause License
#
# Copyright (c) 2023 Pan St
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.

import argparse
from itertools import chain

class Col(list):
    def __init__(self, height: int):
        super().__init__()
        self.extend((' ',) * height)
        self.pos = 0

    def written(self) -> int:
        return self.pos

    def write(self, word: str):
        for i in range(len(word)):
            self[i + self.pos] = word[i]
        self.pos += len(word)

class Cols:
    def __init__(self, height: int):
        self.height = height
        self.cols = [Col(height)]

    @property
    def col(self) -> Col:
        return self.cols[-1]

    def add(self):
        self.cols.append(Col(self.height))

    def write(self, word: str):
        if len(word) + self.col.written() >= self.height:
            self.add()
        self.col.write(word + ' ')

    def __str__(self) -> str:
        # Live and prosper.
        return '\n'.join(map(lambda i: ' '.join(map(lambda j: self.cols[j][i],
                                                    range(len(self.cols)))),
                             range(self.height)))

def to_cols(s: str, height: int) -> Cols:
    words = s.split(' ')
    actual: int = max(chain((height,), map(len, words))) + 1  # Actual height.
    cols = Cols(actual)
    for word in words:
        cols.write(word)
    return cols

def positive(arg: str) -> int:
    n = int(arg)
    if n <= 0:
        raise argparse.ArgumentTypeError(
            "invalid value %r; expected positive integer." % arg
        )
    return n

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--height", help="maximum column height",
                        type=positive, default=15)
    args = parser.parse_args()
    height = args.height
    print(f"{height=}")
    text = input("type text: ")
    cols = to_cols(text, height)
    print(cols)

if __name__ == "__main__":
    main()
    raise SystemExit