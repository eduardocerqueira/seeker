#date: 2023-03-09T16:53:37Z
#url: https://api.github.com/gists/3f049aac0a72df7bda8f8b78e362421a
#owner: https://api.github.com/users/mypy-play

import abc
from typing import Dict, Tuple


class Tensor:
    pass

class StateT:
    pass


class Module(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(
        self, inputs: Tensor, state: StateT
    ) -> Tuple[Tensor, StateT]:
        ...


class CondModule(Module):
    def forward(
        self, inputs: Tensor, state: StateT, *, context: Dict[str, Tensor]
    ) -> Tuple[Tensor, StateT]:
        raise NotImplementedError()
