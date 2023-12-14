#date: 2023-12-14T16:57:52Z
#url: https://api.github.com/gists/301bdc72dbe39b7eca54aea37521aa15
#owner: https://api.github.com/users/nzec

import functools
from collections import defaultdict
import graphviz
import sys

to_trace = {}
def graphme(*args):
    def decorator(f):
        to_trace[f.__name__] = args
        return f
    return decorator

nodes, edges = set(), defaultdict(int)
def trace(frame, event, arg):
    f_name = frame.f_code.co_name
    if f_name in to_trace:
        f_args = ', '.join([f'{x}={y}' for x, y in frame.f_locals.items() if x in to_trace[f_name]])
        c_args = ', '.join([f'{x}={y}' for x, y in frame.f_back.f_locals.items() if x in to_trace[f_name]])
        c_name = frame.f_back.f_code.co_name
        f_node = f'{f_name}({f_args})'
        nodes.add(f_node)
        if c_name != '<module>':
            c_node = f'{c_name}({c_args})'
            edges[(c_node, f_node)] += 1
        # print(f_name, c_name)

sys.settrace(trace)

def view():
    dot = graphviz.Digraph()
    for node in nodes:
        dot.node(str(hash(node)), node, shape='box')
    for (src, dst), num in edges.items():
        dot.edge(str(hash(src)), str(hash(dst)), label=str(num))
    dot.view()
    # print(dot.source)
