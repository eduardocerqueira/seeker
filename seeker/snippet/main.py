#date: 2023-09-18T16:59:44Z
#url: https://api.github.com/gists/b2b5bb3a33dc5d92ba52568bbedb088f
#owner: https://api.github.com/users/mypy-play

from typing import Type, TypeVar

class Animal: ...
class Snake(Animal): ...

T = TypeVar('T', bound=Animal)

def make_animal(animal_type: Type[T]) -> T:  # <-- what should `Type[T]` be?
    return animal_type()

reveal_type(make_animal(Animal))
reveal_type(make_animal(Snake))