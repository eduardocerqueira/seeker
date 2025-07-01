#date: 2025-07-01T16:50:56Z
#url: https://api.github.com/gists/0f182dfd2254afd01198b937b6ad04d6
#owner: https://api.github.com/users/millaker

#!/bin/env python3
import random


class Node:
    def print_paren(self, func):
        print("(", end="")
        func()
        print(")", end="")

    def precedence(self):
        pass

    def print(self):
        pass


class Binary(Node):
    def __init__(self):
        super().__init__()

    def is_leaf(self) -> bool:
        return False


class Add(Binary):
    def __init__(self, lnode: Node, rnode: Node):
        super().__init__()
        self.child: list[Node] = [lnode, rnode]

    def print(self) -> None:
        if self.precedence() > self.child[0].precedence():
            self.print_paren(self.child[0].print)
        else:
            self.child[0].print()
        print(" + ", end="")
        if self.precedence() > self.child[1].precedence():
            self.print_paren(self.child[1].print)
        else:
            self.child[1].print()

    def eval(self) -> int:
        return int(self.child[0].eval() + self.child[1].eval())

    def precedence(self) -> int:
        return 1


class Sub(Binary):
    def __init__(self, lnode: Node, rnode: Node):
        super().__init__()
        self.child: list[Node] = [lnode, rnode]

    def print(self) -> None:
        if self.precedence() > self.child[0].precedence():
            self.print_paren(self.child[0].print)
        else:
            self.child[0].print()
        print(" - ", end="")
        if self.precedence() > self.child[1].precedence():
            self.print_paren(self.child[1].print)
        else:
            self.child[1].print()

    def eval(self) -> int:
        return int(self.child[0].eval() - self.child[1].eval())

    def precedence(self) -> int:
        return 1


class Mul(Binary):
    def __init__(self, lnode: Node, rnode: Node):
        super().__init__()
        self.child: list[Node] = [lnode, rnode]

    def print(self) -> None:
        if self.precedence() > self.child[0].precedence():
            self.print_paren(self.child[0].print)
        else:
            self.child[0].print()
        print(" * ", end="")
        if self.precedence() > self.child[1].precedence():
            self.print_paren(self.child[1].print)
        else:
            self.child[1].print()

    def eval(self) -> int:
        return int(self.child[0].eval() * self.child[1].eval())

    def precedence(self) -> int:
        return 2


class Div(Binary):
    def __init__(self, lnode: Node, rnode: Node):
        super().__init__()
        self.child: list[Node] = [lnode, rnode]

    def print(self) -> None:
        if self.precedence() > self.child[0].precedence():
            self.print_paren(self.child[0].print)
        else:
            self.child[0].print()
        print(" / ", end="")
        if self.precedence() > self.child[1].precedence():
            self.print_paren(self.child[1].print)
        else:
            self.child[1].print()

    def eval(self) -> int:
        return int(self.child[0].eval() / self.child[1].eval())

    def precedence(self) -> int:
        return 2


class Num(Node):
    def __init__(self, val: int):
        self.val = val

    def print(self) -> None:
        print(self.val, end="")

    def eval(self) -> int:
        return int(self.val)

    def precedence(self) -> int:
        return 3


def generator(temp: int, range: int) -> Node:
    """Recursively generates a random mathematical expression tree.

    Args:
        temp (int): Controls the complexity of the tree.
        range_val (int): The upper bound for random numbers.

    Returns:
        Node: The root node of the generated tree.
    """
    if temp == 0 or random.randint(0, temp) == 0:
        return Num(random.randint(0, range))
    else:
        node_types = [Add, Sub, Mul, Div]
        weights = [0.30, 0.30, 0.30, 0.05]
        op = random.choices(population=node_types, weights=weights)[0]
        new_temp = int(temp / 1.5)
        lnode = generator(new_temp, range)
        rnode = generator(new_temp, range)
        return op(lnode, rnode)


help(generator)
a = generator(10, 1000)
a.print()
print(" =", a.eval())
