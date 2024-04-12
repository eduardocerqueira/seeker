#date: 2024-04-12T16:41:28Z
#url: https://api.github.com/gists/219e1dc416565ad42070ea13e4010c75
#owner: https://api.github.com/users/morozov

from abc import ABC, abstractmethod


class Operation(ABC):
    @abstractmethod
    def apply(self, color):
        pass

    @abstractmethod
    def __str__(self):
        pass


class Nop(Operation):
    def apply(self, color):
        return color

    def __str__(self):
        return f"NOP"


class And(Operation):
    def __init__(self, value):
        self.value = value

    def apply(self, color):
        return color & self.value

    def __str__(self):
        return f"AND {self.value}"


class Or(Operation):
    def __init__(self, value):
        self.value = value

    def apply(self, color):
        return color | self.value

    def __str__(self):
        return f"OR {self.value}"


class Xor(Operation):
    def __init__(self, value):
        self.value = value

    def apply(self, color):
        return color ^ self.value

    def __str__(self):
        return f"XOR {self.value}"


class Combo(Operation):
    def __init__(self, *operations):
        self.operations = operations

    def apply(self, color):
        for operation in self.operations:
            color = operation.apply(color)
        return color

    def __str__(self):
        return ", ".join([str(operation) for operation in self.operations])


class ColorPair:
    def __init__(self, color1, color2):
        self.color1 = min(color1, color2)
        self.color2 = max(color1, color2)

    def transform(self, operation):
        return ColorPair(operation.apply(self.color1), operation.apply(self.color2))

    def __eq__(self, other):
        if not isinstance(other, ColorPair):
            return NotImplemented

        return self.color1 == other.color1 and self.color2 == other.color2

    def __str__(self):
        return f"({self.color1}, {self.color2})"


class Transformation:
    colors: ColorPair
    operation: Operation

    def __init__(self, colors, operation):
        self.colors = colors
        self.operation = operation


transformations = [
    Transformation(ColorPair(0, 1), And(1)),
    Transformation(ColorPair(0, 2), And(2)),
    Transformation(ColorPair(0, 3), Combo(Xor(1), And(3))),
    Transformation(ColorPair(0, 4), And(4)),
    Transformation(ColorPair(0, 5), Combo(Xor(1), And(5))),
    Transformation(ColorPair(0, 6), And(6)),
    Transformation(ColorPair(0, 7), Xor(1)),
    Transformation(ColorPair(1, 2), And(3)),
    Transformation(ColorPair(1, 3), Combo(And(3), Or(1))),
    Transformation(ColorPair(1, 4), And(5)),
    Transformation(ColorPair(1, 5), Combo(And(5), Or(1))),
    Transformation(ColorPair(1, 6), Nop()),
    Transformation(ColorPair(1, 7), Or(1)),
    Transformation(ColorPair(2, 3), Combo(Or(2), And(3))),
    Transformation(ColorPair(2, 4), Combo(And(6), Xor(2))),
    Transformation(ColorPair(2, 5), Xor(3)),
    Transformation(ColorPair(2, 6), Combo(And(6), Or(2))),
    Transformation(ColorPair(2, 7), Combo(Xor(1), Or(2))),
    Transformation(ColorPair(3, 4), Xor(2)),
    Transformation(ColorPair(3, 5), Combo(Or(1), Xor(2))),
    Transformation(ColorPair(3, 6), Or(2)),
    Transformation(ColorPair(3, 7), Or(3)),
    Transformation(ColorPair(4, 5), Combo(Or(4), And(5))),
    Transformation(ColorPair(4, 6), Combo(And(6), Or(4))),
    Transformation(ColorPair(4, 7), Combo(Xor(1), Or(4))),
    Transformation(ColorPair(5, 6), Or(4)),
    Transformation(ColorPair(5, 7), Or(5)),
    Transformation(ColorPair(6, 7), Or(6)),
]

default = ColorPair(1, 6)
if __name__ == "__main__":
    for transformation in transformations:
        test = "Pass" if default.transform(transformation.operation) == transformation.colors else "Fail"
        print(f'{transformation.colors}: {transformation.operation} → {test}')
