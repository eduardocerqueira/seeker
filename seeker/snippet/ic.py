#date: 2025-02-04T17:02:10Z
#url: https://api.github.com/gists/99fb0fa6ba7ca69e25b4fc6cdf1081fb
#owner: https://api.github.com/users/LeeeeT

from collections import defaultdict
from dataclasses import dataclass
from typing import NewType


Wire = NewType("Wire", int)


@dataclass(frozen=True)
class Constructor:
	left: Wire
	right: Wire
	principal: Wire


@dataclass(frozen=True)
class Duplicator:
	left: Wire
	right: Wire
	principal: Wire


@dataclass(frozen=True)
class Eraser:
	principal: Wire


@dataclass(frozen=True)
class Free:
	principal: Wire


Cell = Constructor | Duplicator | Eraser | Free


def ports(cell: Cell) -> set[Wire]:
	match cell:
		case Constructor(left, right, principal):
			return {left, right, principal}
		case Duplicator(left, right, principal):
			return {left, right, principal}
		case Eraser(principal):
			return {principal}
		case Free(principal):
			return {principal}


def aux(cell: Cell) -> set[Wire]:
	match cell:
		case Constructor(left, right, _):
			return {left, right}
		case Duplicator(left, right, _):
			return {left, right}
		case Eraser(_):
			return set()
		case Free(_):
			return set()


def rewire(cell: Cell, first: Wire, second: Wire) -> Cell:
	match cell:
		case Constructor(left, right, principal):
			if left == first:
				left = second
			if right == first:
				right = second
			if principal == first:
				principal = second
			return Constructor(left, right, principal)
		case Duplicator(left, right, principal):
			if left == first:
				left = second
			if right == first:
				right = second
			if principal == first:
				principal = second
			return Duplicator(left, right, principal)
		case Eraser(principal):
			if principal == first:
				principal = second
			return Eraser(principal)
		case Free(principal):
			if principal == first:
				principal = second
			return Free(principal)


def print_cell(cell: Cell) -> None:
	match cell:
		case Constructor(left, right, principal):
			print(f"γ({left},{right})→{principal}")
		case Duplicator(left, right, principal):
			print(f"δ({left},{right})→{principal}")
		case Eraser(principal):
			print(f"ε→{principal}")
		case Free(principal):
			print(f"φ→{principal}")


class Net(defaultdict[Wire, set[Cell]]):
	def __init__(self) -> None:
		super().__init__(set)
		self.free_wire = Wire(0)


def cells(net: Net) -> set[Cell]:
	return set[Cell]().union(*net.values())


def new_wire(net: Net) -> Wire:
	wire = net.free_wire
	net.free_wire = Wire(wire + 1)
	return wire


def add(net: Net, cell: Cell) -> None:
	for wire in ports(cell):
		net[wire].add(cell)
	net.free_wire = Wire(max(net.free_wire, max(ports(cell)) + 1))


def remove(net: Net, cell: Cell) -> None:
	for wire in ports(cell):
		net[wire].remove(cell)
		if not net[wire]:
			del net[wire]


def join_wires(net: Net, first: Wire, second: Wire) -> None:
	for cell in net[first].copy():
		remove(net, cell)
		add(net, rewire(cell, first, second))


def interact(net: Net, first: Cell, second: Cell) -> bool:
	match first, second:
		case Constructor(a, b, c), Constructor(d, e, f) if c == f:
			remove(net, first)
			remove(net, second)
			join_wires(net, a, e)
			join_wires(net, b, d)
			return True
		case Constructor(a, b, c), Duplicator(d, e, f) if c == f:
			remove(net, first)
			remove(net, second)
			g = new_wire(net)
			h = new_wire(net)
			i = new_wire(net)
			j = new_wire(net)
			add(net, Duplicator(g, h, a))
			add(net, Duplicator(i, j, b))
			add(net, Constructor(g, i, d))
			add(net, Constructor(h, j, e))
			return True
		case Constructor(a, b, c), Eraser(d) if c == d:
			remove(net, first)
			remove(net, second)
			add(net, Eraser(a))
			add(net, Eraser(b))
			return True
		case Duplicator(a, b, c), Constructor(d, e, f) if c == f:
			remove(net, first)
			remove(net, second)
			g = new_wire(net)
			h = new_wire(net)
			i = new_wire(net)
			j = new_wire(net)
			add(net, Constructor(g, h, a))
			add(net, Constructor(i, j, b))
			add(net, Duplicator(g, i, d))
			add(net, Duplicator(h, j, e))
			return True
		case Duplicator(a, b, c), Duplicator(d, e, f) if c == f:
			remove(net, first)
			remove(net, second)
			join_wires(net, a, d)
			join_wires(net, b, e)
			return True
		case Duplicator(a, b, c), Eraser(d) if c == d:
			remove(net, first)
			remove(net, second)
			add(net, Eraser(a))
			add(net, Eraser(b))
			return True
		case Eraser(a), Constructor(b, c, d) if a == d:
			remove(net, first)
			remove(net, second)
			add(net, Eraser(b))
			add(net, Eraser(c))
			return True
		case Eraser(a), Duplicator(b, c, d) if a == d:
			remove(net, first)
			remove(net, second)
			add(net, Eraser(b))
			add(net, Eraser(c))
			return True
		case Eraser(a), Eraser(b) if a == b:
			remove(net, first)
			remove(net, second)
			return True
		case _:
			return False


def reduce(net: Net) -> None:
	queue = set(net)
	while queue:
		match tuple(net[queue.pop()]):
			case first, second:
				if interact(net, first, second):
					queue.update(aux(first), aux(second))
			case _:
				pass


def print_net(net: Net) -> None:
	for cell in cells(net):
		print_cell(cell)


def multiplexor(net: Net, principal: Wire, *auxes: Wire) -> None:
	match auxes:
		case ():
			add(net, Eraser(principal))
		case (aux,):
			join_wires(net, principal, aux)
		case (first, *rest):
			a = new_wire(net)
			multiplexor(net, a, *rest)
			add(net, Duplicator(first, a, principal))


def main() -> None:
	net = Net()
	a = new_wire(net)
	b = new_wire(net)
	c = new_wire(net)
	d = new_wire(net)
	e = new_wire(net)
	f = new_wire(net)
	g = new_wire(net)
	h = new_wire(net)
	i = new_wire(net)
	multiplexor(net, a, b, c, d, e)
	multiplexor(net, a, f, g, h, i)

	print("Non-reduced:")
	print_net(net)

	print()
	reduce(net)

	print("Reduced:")
	print_net(net)


main()