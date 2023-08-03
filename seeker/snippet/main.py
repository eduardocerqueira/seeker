#date: 2023-08-03T17:02:57Z
#url: https://api.github.com/gists/be50026196853334f8380c5762656f0d
#owner: https://api.github.com/users/mypy-play

from typing import TypeGuard, Any

class ClassA:
    pass

class ClassB:
    def classB_method(self):
        print("class B!")

class ClassC(ClassA, ClassB):
    pass

def guard(item: Any) -> TypeGuard[ClassA]:
    return isinstance(item, ClassA)


dog = ClassC()

assert guard(dog)

dog.classB_method()

