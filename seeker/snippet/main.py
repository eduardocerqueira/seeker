#date: 2022-08-31T16:55:02Z
#url: https://api.github.com/gists/2f09c0e569aa5e5120edbc22ad74aa84
#owner: https://api.github.com/users/mypy-play

from typing import NoReturn

def assert_never(x: NoReturn) -> NoReturn:
    raise AssertionError(f"Invalid value: {x!r}")
    
class BaseAnimal:
    pass

class Dog(BaseAnimal):
    pass

class Cat(BaseAnimal):
    pass

Animal = Dog | Cat

x: Animal = Dog()

if isinstance(x, Dog):
    print("dog")
# elif isinstance(x, Cat):
#     print("cat")
else:
    assert_never(x)
    