#date: 2025-03-31T16:47:04Z
#url: https://api.github.com/gists/b111569d7dee9908673e66082f3d6833
#owner: https://api.github.com/users/mypy-play

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, TypeVar, Generic


@dataclasses.dataclass
class ActionContext:
    data: str
    
    
@dataclasses.dataclass
class CmdActionContext(ActionContext):
    data_cmd: str



_DecoratedPRG = Callable[["ActionContext"], None]
_DecoratedCMD = Callable[["CmdActionContext"], None]
_DecoratedT = TypeVar("_DecoratedT", _DecoratedPRG, _DecoratedCMD)


@dataclasses.dataclass
class ActionDefinition(Generic[_DecoratedT]):
    action: _DecoratedPRG
    action_kwargs: dict[str, Any]


action_defs: dict[str, ActionDefinition] = {}


def _register_action(
    action_name: str, **kwargs: str
) -> Callable[[_DecoratedT], _DecoratedT]:
    """
    Decorator function to register an action.

    Args:
        action_name: The name of the action.
        kwargs: Optional arguments for the action (gets passed to controller).
    """

    def _register_action(obj: _DecoratedT) -> _DecoratedT:
        action_defs[action_name] = ActionDefinition(action=obj, action_kwargs=kwargs)
        return obj

    return _register_action


def register_cmd(
    action_name: str, **kwargs: str
) -> Callable[[_DecoratedCMD], _DecoratedCMD]:
    return _register_action(action_name, **kwargs)

def register_prg(
    action_name: str, **kwargs: str
) -> Callable[[_DecoratedPRG], _DecoratedPRG]:
    return _register_action(action_name, **kwargs)