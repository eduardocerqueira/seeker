#date: 2022-02-01T16:58:21Z
#url: https://api.github.com/gists/a2a1c1a1b81a35afddbbe96551c6e9f7
#owner: https://api.github.com/users/hartwork

# Copyright (c) 2022 Sebastian Pipping <sebastian@pipping.org>
# Licensed under the Apache license version 2.0
#
# Needs Python >=3.6
# Version 2022-02-01 17:50 UTC+1

_full_block_char = '\u2588'
_ansi_escape = '\u001b'
_ansi_reset = f'{_ansi_escape}[0m'
_demo_text = 2 * _full_block_char


def _3bit_ansi_foreground(i, bright):
    assert 0 <= i < 8
    i += 30
    bright_mod = ';1' if bright else ''
    return f'{_ansi_escape}[{i}{bright_mod}m'

def _4bit_ansi_foreground(i):
    assert 0 <= i < 8
    i += 90
    return f'{_ansi_escape}[{i}m'

def _8bit_ansi_foreground(i):
    assert i <= i <= 255
    return f'{_ansi_escape}[38;5;{i}m'


print('3bit')
for bright in (False, True):
    for i in range(8):
        print(f"{_3bit_ansi_foreground(i, bright=bright)}{_demo_text}{_ansi_reset}", end="")
    print()
print()


print('4bit')
for i in range(8):
    print(f"{_3bit_ansi_foreground(i, bright=False)}{_demo_text}{_ansi_reset}", end="")
print()
for i in range(8):
    print(f"{_4bit_ansi_foreground(i)}{_demo_text}{_ansi_reset}", end="")
print()
print()


print('8bit')
for i in range(256):
    print(f"{_8bit_ansi_foreground(i)}{_demo_text}{_ansi_reset}", end="")
    if i % 8 == 7:
        print()
print()
