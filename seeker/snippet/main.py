#date: 2023-03-03T16:46:22Z
#url: https://api.github.com/gists/7d055f2c37ce94c8042fb35298091ddc
#owner: https://api.github.com/users/mypy-play

from typing import Callable, List, Any, Dict, Type, TypeVar, Optional, Generic, cast, Protocol,Union
from collections.abc import MutableSequence, Sequence
from decimal import Decimal
from dataclasses import dataclass, fields
from abc import ABC, abstractmethod



DATA = TypeVar("DATA")


class CommandValidator(ABC, Generic[DATA]):
    def __init__(self, aggregate: str) -> None:
        self.aggregate = aggregate
        
    @abstractmethod
    def validate(self, data: DATA) -> None:
        ...



class FutureCancellationValidatorProtocol(Protocol):
    canceled_date: str
    
        

class MProtocol(Protocol):
    b: str
    slight: int



class FutureCancellationValidator(CommandValidator[FutureCancellationValidatorProtocol]):
    def validate(self, data: FutureCancellationValidatorProtocol) -> None:
        ...

        
class BhadsgfCancellationValidator(CommandValidator[MProtocol]):
    def validate(self, data: MProtocol) -> None:
        ...
        

class NoProtocol(Protocol):
    ...


class BlahValidator(CommandValidator[Any]):
    def validate(self, data: Any) -> None:
        ...
        
        
class OptionCancelRequest:
    canceled_date: str
    something: int


class CancelValidator:
    def __init__(self, aggregate: str) -> None:
        self.aggregate = aggregate

    def validate(self, request: OptionCancelRequest) -> None:
        CANCEL_VALIDATORS: List[Union[Type[FutureCancellationValidator], Type[BhadsgfCancellationValidator], Type[BlahValidator]]] = [
            FutureCancellationValidator,
            BhadsgfCancellationValidator,
            BlahValidator
        ]
        # BhadsgfCancellationValidator(self.aggregate).validate(request)
        # FutureCancellationValidator(self.aggregate).validate(request)
        # BlahValidator(self.aggregate).validate(request)

        for validator in CANCEL_VALIDATORS:
            validator(self.aggregate).validate(request)
