#date: 2021-12-28T17:17:09Z
#url: https://api.github.com/gists/61e0e5632bac60e8d8c0016ed92da82d
#owner: https://api.github.com/users/llimllib

# modified from https://github.com/pallets/click/blob/051d57cef4ce59212dc1175ad4550743bf47d840/src/click/_termui_impl.py
import contextlib
import codecs
import os
import sys
import tty
import termios
import typing as t


def _translate_ch_to_exc(ch: str) -> t.Optional[BaseException]:
    if ch == "\x03":
        raise KeyboardInterrupt()

    if ch == "\x04":  # Unix-like, Ctrl+D
        raise EOFError()

    return None


def is_ascii_encoding(encoding: str) -> bool:
    """Checks if a given encoding is ascii."""
    try:
        return codecs.lookup(encoding).name == "ascii"
    except LookupError:
        return False


def get_best_encoding(stream: t.IO) -> str:
    """Returns the default stream encoding if not found."""
    rv = getattr(stream, "encoding", None) or sys.getdefaultencoding()
    if is_ascii_encoding(rv):
        return "utf-8"
    return rv


@contextlib.contextmanager
def raw_terminal() -> t.Iterator[int]:
    f: t.Optional[t.TextIO]
    fd: int

    if not sys.stdin.isatty():
        f = open("/dev/tty")
        fd = f.fileno()
    else:
        fd = sys.stdin.fileno()
        f = None

    try:
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            yield fd
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            sys.stdout.flush()

            if f is not None:
                f.close()
    except termios.error:
        pass


# this won't work in windows because I only copied the unix version
def getchar(echo: bool) -> str:
    with raw_terminal() as fd:
        ch = os.read(fd, 32).decode(get_best_encoding(sys.stdin), "replace")

        if echo and sys.stdout.isatty():
            sys.stdout.write(ch)

        _translate_ch_to_exc(ch)
        return ch


if __name__ == "__main__":
    import json

    while True:
        ch = getchar(False)
        print(json.dumps(ch))
        if ch == "\x1b[A":
            print("up")
        if ch == "\x1b[B":
            print("down")
        if ch == "\x1b[B":
            print("right")
        if ch == "\x1b[D":
            print("left")
