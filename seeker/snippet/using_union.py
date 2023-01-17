#date: 2023-01-17T16:55:05Z
#url: https://api.github.com/gists/6cb4c9445474bb6a718a81fd28455e0d
#owner: https://api.github.com/users/inspiralpatterns

from typing import Union

class FamilyHeight():
    def __init__(self):
        self.family_heights: List[Union[int, float]] = []

    def push(self, height: Union[int, float]):
        self.family_heights.append(height)

    def pop(self) -> Union[int, float]:
        return self.family_heights.pop()


family_heights_cm = FamilyHeight()
family_heights_cm.push(187)
family_heights_cm.push(6.0)  # No type checked


def return_first(container: List[Union[int, float]) -> Union[int, float]:
    return container[0]
    
my_height: int = return_first(family_heights_cm)   # Type Error