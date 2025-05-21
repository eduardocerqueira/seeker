#date: 2025-05-21T16:54:44Z
#url: https://api.github.com/gists/2d005c207f918b60b954ee19753d34e8
#owner: https://api.github.com/users/dancergraham

"""
Variable history
This module adds a way to track the history of variables.
It does this by adding a `sys.settrace` hook that captures the values of
variables as they are set, recording a copy of `globals()` at each point.
Ideas by Pedro Diniz (@pedrowd)
First Implementation by Pedro Diniz (@pedrowd) and Daniel Diniz (@devdanzin)
This Implementation by Graham Knapp and GitHub Copilot (GPT4.1)
"""
import sys
from types import FrameType
from typing import Any, List

MEMORY: List[dict[str, Any]] = []
SENTINEL = object()

def trace_assignments(frame: FrameType, event: str, arg):
    if event == "line":
        # Record a copy of globals after each line
        MEMORY.append(frame.f_globals.copy())
    return trace_assignments

def show_history(names):
    """Show the history of variables in `names`."""
    for name in names:
        last = SENTINEL
        for entry in MEMORY:
            if name in entry and last is SENTINEL:
                print(name, "initially set to", entry[name])
                last = entry[name]
            elif name in entry and last is not SENTINEL and entry[name] != last:
                print(name, "changed from", last, "to", entry[name])
                last = entry[name]

# Set up tracing
sys.settrace(trace_assignments)

# Example assignments (these will be recorded)
a = 10
b = "blah"
c = True
d = 3.1415926535
a = 11
b = "blahba"
c = False
d = 0.00001

# Stop tracing before showing history to avoid recording further changes
sys.settrace(None)

show_history(["a", "b", "c", "d"])