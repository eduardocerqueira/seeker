#date: 2025-08-06T17:12:57Z
#url: https://api.github.com/gists/62b3d729e2bed47ee673aae2fd46a916
#owner: https://api.github.com/users/pschanely

import dataclasses
from typing import List

@dataclasses.dataclass
class AverageableQueue:
    '''
    A queue of numbers with a O(1) average() operation.
    inv: self._total == sum(self._values)
    '''
    _values: List[int]
    _total: int

    def push(self, val: int):
        self._values.append(val)
        self._total += val

    def pop(self) -> int:
        ''' pre: len(self._values) > 0 '''
        val = self._values.pop(0)
        # Oops. We are forgetting to do something here.
        return val

    def average(self) -> float:
        ''' pre: self._values '''
        return self._total / len(self._values)
