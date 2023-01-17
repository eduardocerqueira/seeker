#date: 2023-01-17T17:02:03Z
#url: https://api.github.com/gists/cc2149d9aad1f64400cf30ea5cf3a1c0
#owner: https://api.github.com/users/inspiralpatterns

from typing import Any

class FamilyHeight():
    def __init__(self):
        self.family_heights: List[Any] = []

    def push(self, height: Any):
        self.family_heights.append(height)

    def pop(self) -> Any:
        return self.family_heights.pop()


family_heights_cm = FamilyHeight()
family_heights_cm.push(187)


def return_first(container: List[Any]) -> Any:
    return container[0]
    
my_height: int = return_first(family_heights_cm)   # OK