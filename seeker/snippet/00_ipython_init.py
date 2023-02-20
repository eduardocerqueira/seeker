#date: 2023-02-20T16:57:46Z
#url: https://api.github.com/gists/a388976a3ee3c80d45792c3adb9b876d
#owner: https://api.github.com/users/swyckoff

import gzip
import itertools
import json
import os
import pickle
import random
import re
import sys
import types
from collections import Counter, defaultdict, namedtuple


def import_module(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        print(
            f"IMPORTERROR: Failed to import {module_name}. Please make sure it is installed."
        )


try:
    rich = import_module("rich")
    from rich import print
except ImportError:
    ...

np = import_module("numpy")
pd = import_module("pandas")

ipdb = import_module("ipdb")


def imports():
    for _, val in globals().items():
        if isinstance(val, types.ModuleType):
            yield val.__name__


print('Successfully imported: [{:s}]'.format(', '.join(
    sorted(
        set([
            '"{:s}"'.format(e) for e in imports()
            if '__' not in e and 'types' not in e
        ])))))
print('Reminders:')

reminders = [
    ('object??:', 'display the source.'), ('%edit', 'write a long function.'),
    ('os.*dir*?', 'wildcard search.'), ('%debug', 'post-mortem debugging.'),
    ('!pwd', 'run shell commands.'), ('%cd', 'move around file system.'),
    ('%rerun ~1/', 'rerun all commands from previous session.'),
    ('_1', 'use output of cell 1, or _n.'),
    ('%pastebin 1-5', 'create pastbin of cells 1-5.'),
    ('breakpoint()', 'export PYTHONBREAKPOINT=ipdb.set_trace.'),
    ('%%ruby', 'run code in ruby, or language.'),
    ('%save filename.py 1-4', 'save code.'),
    ('%paste', 'remove > from clipboard code.'),
    ('%whos', 'list variables in session.'),
    ('%ipdb', 'Set ipdb as the debugger'),
    ('%autoreload 2',
     'reloads modules automatically before entering the execution of a code block'
     )
]

for command, description in reminders:
    print(f'"{command}": {description}')
