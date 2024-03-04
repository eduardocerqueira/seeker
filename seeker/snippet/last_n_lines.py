#date: 2024-03-04T17:12:05Z
#url: https://api.github.com/gists/21e77b42cf85155c42054c7ee5681e43
#owner: https://api.github.com/users/PaulisMatrix

import os
from collections import deque
from collections.abc import Iterator, Sequence
from typing import Final, Protocol


class SeekableBytesFile(Protocol):
    def seek(self, position: int, whence: int = ..., /) -> int: ...
    def read(self, amount: int, /) -> bytes: ...


BUFFER_SIZE: Final = 8192


def iter_lines_backwards(file: SeekableBytesFile) -> Iterator[str]:
    """Lazily iterate through the lines of a file in reverse order.

    This function draws on the Stack Overflow answer
    https://stackoverflow.com/a/23646049/13990016,
    originally by srohde.
    """
    # Move the cursor to the end of the file
    previous_position = cursor_position = file.seek(0, os.SEEK_END)
    leftover = b""
    first_iteration = True

    # Iteratively move the cursor backwards through the file,
    # reading a fixed chunk at a time
    while cursor_position > 0:
        cursor_position = max(0, cursor_position - BUFFER_SIZE)
        file.seek(cursor_position)

        chunk_size = previous_position - cursor_position
        chunk = file.read(chunk_size)
        chunk_lines = chunk.splitlines()

        # We'll depend on this invariant for much of the rest of this function:
        assert chunk_lines, "`chunk_lines` should always be non-empty if `cursor > 0`"

        # Discard a trailing newline from the end of the file
        if first_iteration:
            first_iteration = False
            if chunk_lines[-1].endswith(b"\n"):
                chunk_lines[-1] = chunk_lines[-1][:-1]

        # If `leftover` is truthy, it means that the previous chunk
        # began halfway through a line;
        # we'll need to add the previous chunk's
        # first line onto this chunk's last line to recreate a complete line
        elif leftover:
            chunk_lines[-1] += leftover

        # `first_line-this_chunk` will either be `b""`,
        # meaning the chunk started with a newline separator
        # (which we can safely discard when processing the next chunk),
        # or it will be a non-empty bytes sequence,
        # indicating that this chunk started halfway through a line.
        #
        # N.B. `.pop(0)` is, in general, inefficient if you're using a list.
        # To address that, we could in theory convert `chunk_lines` into a deque,
        # and then use the `popleft` method.
        # In practice, however, `chunk_lines` here is always likely
        # to be a very small list, meaning the cost of creating the deque
        # outweighs the inefficiency of doing `.pop(0)` on a list.
        first_line_this_chunk = chunk_lines.pop(0)

        yield from map(bytes.decode, reversed(chunk_lines))

        # We've processed all the lines in this chunk;
        # now prepare for the next chunk:
        leftover = first_line_this_chunk
        previous_position = cursor_position

    if leftover:
        yield leftover.decode()


def last_n_lines_of_file(filename: str, *, n: int) -> Sequence[str]:
    """Return the last `n` lines of an unopened file."""
    lines = deque[str]()
    with open(filename, "rb") as f:
        backward_lines_iterator = iter_lines_backwards(f)
        while len(lines) < n:
            try:
                next_line = next(backward_lines_iterator)
            except StopIteration:
                return lines
            else:
                lines.appendleft(next_line)
    return lines


if __name__ == "__main__":
    import sys
    for line in last_n_lines_of_file(sys.argv[1], n=int(sys.argv[2])):
        print(line)
