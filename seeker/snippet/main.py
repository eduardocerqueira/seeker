#date: 2024-10-31T16:48:35Z
#url: https://api.github.com/gists/94b24135fcdd8d6a1c021270eb622c12
#owner: https://api.github.com/users/mypy-play

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, NewType, Self, cast
from uuid import UUID, uuid4

_Key = TypeVar("_Key", contravariant=True)
_Model = TypeVar("_Model")


class InnerProtocol(Protocol[_Model, _Key]):
    def get(self, id: _Key) -> _Model | None: ...
    def add(self, input_model: _Model) -> _Model: ...


Key = TypeVar("Key", contravariant=True)
Model = TypeVar("Model")

class InnerClass(Generic[Model, Key]):
    model: type[Model]

    def get(self, id: Key) -> Model | None:
        return None

    def add(self, model: Model) -> Model:
        return model

Id = NewType("Id", UUID)

@dataclass
class ModelData:
    id: Id
    width: int
    height: int


class ModelProtocol(InnerProtocol[ModelData, Id]):
    pass


class ModelClass(InnerClass[ModelData, Id]):
    model = ModelData


class OuterProtocol(Protocol):
    @property
    def model(self) -> ModelProtocol: ...


class OuterClass:
    def __init__(self) -> None:
        self.model = ModelClass()


def test_function(id: Id, *, uow: OuterClass) -> ModelData | None:
    return uow.model.get(id)


if __name__ == "__main__":
    uow = OuterClass()
    _id = cast(Id, uuid4())
    widget = test_function(_id, uow=uow)
