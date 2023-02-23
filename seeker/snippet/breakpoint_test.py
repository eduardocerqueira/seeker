#date: 2023-02-23T17:09:39Z
#url: https://api.github.com/gists/da7588db3c883dc2006a727a10e00ca5
#owner: https://api.github.com/users/markshannon

import timeit
import pdb
import dis
import sys

def foo():
    for i in range(100_000):
        if i == 50_000:
            pass # Put breakpoint here

print("No debug")
    
print(timeit.timeit(foo, number = 100))

DEBUGGER_ID = 0

if sys.version_info.minor >= 12:
    print("PEP 669")
    
    break_count = 0

    def pep_669_breakpoint(code, line):
        global break_count
        if line == 9 and code.co_name == "foo":
            break_count += 1
        else:
            return sys.monitoring.DISABLE
        
    sys.monitoring.register_callback(DEBUGGER_ID, sys.monitoring.events.LINE, pep_669_breakpoint)
    sys.monitoring.set_events(DEBUGGER_ID, sys.monitoring.events.LINE)

    print(timeit.timeit(foo, number = 100))

    sys.monitoring.set_events(DEBUGGER_ID, 0)
        
    print("Break point hit", break_count, "times")


break_count = 0
#Use sys.settrace, this is about as fast as a sys.settrace debugger can be if written in Python.

print("sys.settrace")

def sys_settrace_breakpoint(frame, event, arg):
    global break_count
    if event == "line" and frame.f_code.co_name == "foo" and frame.f_lineno == 9:
        break_count += 1
    return sys_settrace_breakpoint
        
sys.settrace(sys_settrace_breakpoint)

print(timeit.timeit(foo, number = 100))

sys.settrace(None)


print("Break point hit", break_count, "times")