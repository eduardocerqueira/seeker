#date: 2025-12-22T17:01:09Z
#url: https://api.github.com/gists/8080f806248212447b6577e93925e3e1
#owner: https://api.github.com/users/johnjohnlin

#!/usr/bin/env python
import math
import json

class AstNode:
	def __init__(self, description, value, children=None):
		self.value = value
		self.description = description
		self.children = children if children is not None else []

	@staticmethod
	def Const(description, value):
		return AstNode(description, value)

	@staticmethod
	def Sum(description, *nodes):
		val = sum([node.value for node in nodes])
		children = list(nodes)
		return AstNode(description + " = sum of:", val, children)

	@staticmethod
	def Mul(description, *nodes):
		val = math.prod([node.value for node in nodes])
		children = list(nodes)
		return AstNode(description + " = product of:", val, children)

	@staticmethod
	def Max(description, *nodes):
		val = max([node.value for node in nodes])
		children = list(nodes)
		return AstNode(description + " = max of:", val, children)

	def __add__(self, other):
		return AstNode.Sum(self.description, self, other)

	def __mul__(self, other):
		return AstNode.Mul(self.description, self, other)

	def ToDict(self):
		ret = {
			'name': self.description,
			'value': self.value,
		}
		if len(self.children) > 0:
			ret['children'] = [child.ToDict() for child in self.children]
		return ret

def main():
	expr = AstNode.Sum("Final result",
		AstNode.Mul("Something",
			AstNode.Const("One Hundred", 100),
			AstNode.Const("Magic factor (0.1)", 0.1)
		),
		AstNode.Max("Something",
			AstNode.Const("One", 1),
			AstNode.Const("Three", 3)
		)
	)

	print("JSON:")
	print(json.dumps(expr.ToDict(), indent=2))

if __name__ == "__main__":
	main()