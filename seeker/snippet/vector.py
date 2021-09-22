#date: 2021-09-22T17:07:31Z
#url: https://api.github.com/gists/cfd7979de8ab6f17f8fec144f0882a3d
#owner: https://api.github.com/users/RaphaelGoutmann

# vector.py

class Vector2:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __repr__(self):                             # for debugging
        return f"Vector2({self.x}, {self.y})"

    def __str__(self):                              # for printing
        return f"Vector2({self.x}, {self.y})"
    
    # equal

    def __eq__(self, other):

        if not isinstance(other, Vector2):
            return False

        return ((self.x == other.x) and (self.y == other.y))

    # addition

    def __add__(self, other):

        if isinstance(other, Vector2):
            x = self.x + other.x
            y = self.y + other.y
        else:
            x = self.x + other
            y = self.y + other

        return Vector2(x, y)

    __radd__ = __add__

    # substraction 

    def __sub__(self, other):

        if isinstance(other, Vector2):
            x = self.x - other.x
            y = self.y - other.y
        else:
            x = self.x - other
            y = self.y - other

        return Vector2(x, y)

    def __rsub__(self, other):

        if isinstance(other, Vector2):
            x = other.x - self.x
            y = other.y - self.y
        else:
            x = other - self.x
            y = other - self.y

        return Vector2(x, y)

    # multiplication

    def __mul__(self, other):

        if isinstance(other, Vector2):
            x = self.x * other.x
            y = self.y * other.y
        else:
            x = self.x * other
            y = self.y * other

        return Vector2(x, y)

    __rmul__ = __mul__

    # division

    def __truediv__(self, other):

        if isinstance(other, Vector2):
            x = self.x / other.x
            y = self.y / other.y
        else:
            x = self.x / other
            y = self.y / other

        return Vector2(x, y)

    def __rtruediv__(self, other):

        if isinstance(other, Vector2):
            x = other.x / self.x
            y = other.y / self.y
        else:
            x = other / self.x
            y = other / self.y

        return Vector2(x, y)

    # utility

    def mean(self) -> float:
        return ( (self.x + self.y) / 2.0 )

    def isnull(self) -> bool():
        return ((self.x == 0) and (self.y == 0))

    # up, down, right, left

    def up(self) -> Vector2:
        return Vector2(0, 1)

    def down(self) -> Vector2:
        return Vector2(0, -1)

    def right(self) -> Vector2:
        return Vector2(1, 0)
    
    def left(self) -> Vector2:
        return Vector2(-1, 0)
