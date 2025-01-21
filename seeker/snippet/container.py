#date: 2025-01-21T17:02:06Z
#url: https://api.github.com/gists/0c063c404c523d44d77cb5829abb978f
#owner: https://api.github.com/users/ilya-4real

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar
from uuid import UUID, uuid4


@dataclass
class Entity(ABC):
    oid: UUID = field(default_factory=uuid4, kw_only=True)

ET = TypeVar("ET", bound=Entity)

@dataclass
class Product(Entity):
    is_visible: bool
    quantity: int

class Specification[ET](ABC):
    @abstractmethod 
    def is_satisfied_by(self, candidate: ET) -> bool:
        raise NotImplementedError
    
    def __or__(self, other: "Specification") -> "Specification":
        return OrSpecification(self,other)

    def __and__(self, other: "Specification") -> "Specification":
        return AndSpecification(self,other)

    def __invert__(self) -> "Specification":
        return NotSpecification(self)

class OrSpecification(Specification[ET]):
    def __init__(self, spec_a: Specification, spec_b: Specification) -> None:
        self._spec_a = spec_a
        self._spec_b = spec_b

    def is_satisfied_by(self, candidate: ET) -> bool:
        return self._spec_a.is_satisfied_by(candidate) or self._spec_b.is_satisfied_by(candidate)

class AndSpecification(Specification[ET]):
    def __init__(self, spec_a: Specification, spec_b: Specification) -> None:
        self._spec_a = spec_a
        self._spec_b = spec_b

    def is_satisfied_by(self, candidate: ET) -> bool:
        return self._spec_a.is_satisfied_by(candidate) and self._spec_b.is_satisfied_by(candidate)

class NotSpecification(Specification[ET]):
    def __init__(self, spec: Specification) -> None:
        self._spec = spec
    
    def is_satisfied_by(self, candidate: ET) -> bool:
        return not self._spec.is_satisfied_by(candidate)


class IsVisibleSpec(Specification[Product]):
    def is_satisfied_by(self, candidate: Product) -> bool:
        return candidate.is_visible

class IsAvailable(Specification[Product]):
    def is_satisfied_by(self, candidate: Product) -> bool:
        return candidate.quantity > 0