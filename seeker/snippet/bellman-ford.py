#date: 2024-06-07T16:58:46Z
#url: https://api.github.com/gists/f2c696004e3f3fafa54379c9e96de564
#owner: https://api.github.com/users/i-am-unknown-81514525

from __future__ import annotations

import decimal
from typing import Union, Optional, Mapping


class Node:
    def __init__(self, name: str, connection: Optional[Mapping[Union[str, Node], Union[float, int, decimal.Decimal]]]):
        self.name = name
        self.connection: dict[str, Union[float, int, decimal.Decimal]] = \
            {
                k.name if isinstance(k, Node) else k: v
                for k, v in connection.items()
            } if connection else {}

    def __getitem__(self, name: str) -> Union[float, int, decimal.Decimal]:
        return self.connection.get(name, float('inf'))

    def copy(self) -> CalcNode:
        return CalcNode(self.name, self.connection.copy())

class CalcNode(Node):
    value: Union[float, int, decimal.Decimal]

    def __init__(self, *args, **kwargs):
        self.value = float('inf')
        self.counted = 0
        super().__init__(*args, **kwargs)


# node_dict: dict[str, dict[str, int]] = {
#     "A": {'B': 5, 'C': 35, 'D': 40},
#     "B": {"D":20, "E": 25},
#     "C": {"E": 30, "F": 30},
#     "D": {"F": 20},
#     "E": {"F": 25},
#     "F": {}
# }

# node_dict: dict[str, dict[str, int]] = {
#     "A": {'B': 5, 'C': 35, 'D': 40},
#     "B": {"D":20, "E": 25},
#     "C": {"E": -30, "F": 30},
#     "D": {"F": 20},
#     "E": {"F": 25},
#     "F": {}
# }

node_dict: dict[str, dict[str, int]] = {
    "A": {"B": 1},
    "B": {"C": -1},
    "C": {"D": -1},
    "D": {"E": 1, "B": -1},
    "E": {}
}

node_list = [Node(k, v) for k,v in node_dict.items()]

def calculation(initial: str, list_node: list[Node]):
    predecessor: tuple[str, dict[str, tuple[str, list[str]]]] = (initial, {}) # (initial, {end: (start, stack)})
    list_node: dict[str, CalcNode] = {node.name: node.copy() for node in list_node}
    # Check
    used = {initial, }
    for node in list_node.values():
        used = used.union(node.connection)
    if sorted(list(used)) != sorted(list(list_node)):
        raise ValueError('Have unreachable node or path to non-existent node')
    for node in list_node.values():
        if node.name == initial:
            node.value = 0
    modified = True
    while modified:
        checked = []
        modified = False
        current = initial
        current_node = list_node[current]
        changed = []
        while True:
            for exit, distance in current_node.connection.items():
                if current_node.value + distance < list_node[exit].value:
                    if (current, exit) not in list(zip(predecessor[1].get(current, (current, []))[1], predecessor[1].get(current, (current, []))[1][1:])):
                        list_node[exit].value = current_node.value + distance
                        list_node[exit].counted += 1
                        predecessor[1][exit] = (current, predecessor[1].get(current, (current, []))[1] + [current])
                        modified = True
            checked.append(current)
            if current in changed:
                changed.remove(current)
            if [name for name in changed if name not in checked]:
                current = [name for name in changed if name not in checked][0]
            else:
                if checked == list(list_node):
                    break
                current = [node_name for node_name in list_node if node_name not in checked][0]
            current_node = list_node[current]
    return predecessor

result = calculation('A', node_list)
while True:
    dest = input('>')
    print(' -> '.join(result[1].get(dest, (0, []))[1]))