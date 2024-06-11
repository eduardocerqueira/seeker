#date: 2024-06-11T16:55:38Z
#url: https://api.github.com/gists/d5233c18cc8b65effad88a0a3cfb45ca
#owner: https://api.github.com/users/kcze

from typing import Optional


class Node:
    def __init__(self):
        self.data = None
        self.inputs: list["Node"] = []
        self.outputs: list["Node"] = []

    def execute(self) -> Optional["Node"]:
        raise NotImplementedError("This method should be overridden by subclasses")

    def set(self, *, inputs: list["Node"], outputs: list["Node"]):
        self.inputs = inputs
        self.outputs = outputs


class VarNode(Node):
    def __init__(self, value):
        super().__init__()
        self.data = value

    # VarNode just sets output to its data
    def execute(self) -> Optional["Node"]:
        self.outputs[0].data = self.data
        return self.outputs[0]


class AddOneNode(Node):
    # AddOneNode adds one to its data and passes it to the output
    def execute(self) -> Optional["Node"]:
        if self.data is None:
            return None
        self.outputs[0].data = self.data + 1
        return self.outputs[0]


class LoopNode(Node):
    def execute(self) -> Optional["Node"]:
        if self.data is None:
            return None
        elif self.data < 3:
            self.outputs[0].data = self.data
            return self.outputs[0]
        else:
            self.outputs[1].data = self.data
            return self.outputs[1]


class PrintNode(Node):
    def execute(self) -> Optional["Node"]:
        if self.data is None:
            return None
        print(self.data)
        return Finish()


class Finish(Node): ...


# Create nodes
print_node = PrintNode()
add_node = AddOneNode()
loop_node = LoopNode()
var_node = VarNode(0)

var_node.set(inputs=[], outputs=[add_node])
add_node.set(inputs=[var_node, loop_node], outputs=[loop_node])
loop_node.set(inputs=[add_node], outputs=[add_node, print_node])
print_node.set(inputs=[loop_node], outputs=[])


def execute(node: Node):
    # We need stack when we trace backwards to not make cycles
    stack: list[Node] = [node]

    while stack:
        print(f"Stack: {[node.__class__.__name__ for node in stack]}")
        current_node = stack[-1]
        print(f"  Current: {current_node.__class__.__name__}: {current_node.data}")
        result = current_node.execute()
        # Finish node is returned when the execution is done
        if isinstance(result, Finish):
            break
        # If we got a result, we can clear the stack
        # because we don't need to trace backwards
        if result is not None:
            stack.clear()
            stack.append(result)
            continue
        # Result is None when the node is not ready to execute
        # i.e. the dependencies are not resolved
        # stack.append(current_node)
        for input_node in current_node.inputs:
            if input_node not in stack:
                stack.append(input_node)


execute(print_node)

"""
Output:
Stack: ['PrintNode']
  Current: PrintNode: None
Stack: ['PrintNode', 'LoopNode']
  Current: LoopNode: None
Stack: ['PrintNode', 'LoopNode', 'AddOneNode']
  Current: AddOneNode: None
Stack: ['PrintNode', 'LoopNode', 'AddOneNode', 'VarNode']
  Current: VarNode: 0
Stack: ['AddOneNode']
  Current: AddOneNode: 0
Stack: ['LoopNode']
  Current: LoopNode: 1
Stack: ['AddOneNode']
  Current: AddOneNode: 1
Stack: ['LoopNode']
  Current: LoopNode: 2
Stack: ['AddOneNode']
  Current: AddOneNode: 2
Stack: ['LoopNode']
  Current: LoopNode: 3
Stack: ['PrintNode']
  Current: PrintNode: 3
3
"""