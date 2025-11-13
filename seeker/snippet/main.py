#date: 2025-11-13T17:00:15Z
#url: https://api.github.com/gists/7e51e494b7eb9cb169d2f35f9d59c8e6
#owner: https://api.github.com/users/mypy-play

from typing import Callable, Generic, Literal, Protocol, Type, overload
from typing_extensions import ParamSpec

P = ParamSpec('P')

class Shape[**P](Protocol):
    def __init__(self, *args: P.args, **kwargs: P.kwargs) -> None: ...
    def area(self) -> float: ...
    def perimeter(self) -> float: ...

SHAPE_REGISTRY: dict[str, type[Shape]] = {}

def register_shape(name: str, cls: type[Shape]) -> None:
    SHAPE_REGISTRY[name] = cls

@overload
def create_shape(kind: Literal["circle"], *, radius: float) -> Circle: ...

@overload
def create_shape(kind: Literal["square"], *, side: float) -> Square: ...

def create_shape(kind: str, *args: P.args, **kwargs: P.kwargs) -> Shape[P]:
    cls: Type[Shape[P]] = SHAPE_REGISTRY[kind]
    return cls(*args, **kwargs)
    
class Circle:
    def __init__(self, radius: float):
        self.radius = radius
    def area(self) -> float: return 3.14 * self.radius ** 2
    def perimeter(self) -> float: return 2 * 3.14 * self.radius

class Square:
    def __init__(self, side: float):
        self.side = side
    def area(self) -> float: return self.side ** 2
    def perimeter(self) -> float: return 4 * self.side


register_shape("circle", Circle)
register_shape("square", Square)

circle = create_shape("circle", radius=5.0)
reveal_type(circle)  # note: Revealed type is "__main__.Circle"