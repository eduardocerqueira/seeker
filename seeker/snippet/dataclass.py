#date: 2023-03-29T17:29:53Z
#url: https://api.github.com/gists/4a624924886bcff3058b2713cf1257e4
#owner: https://api.github.com/users/jelc53

from typing import Generic, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

A = TypeVar("A")  # type variable named "A"

class Distribution(ABC, Generic[A]):  # abstract base class (interface)
    @abstractmethod
    def sample(self) -> A:
        pass
    
    def sample_n(self, n: int) -> Sequence[A]:
        return [self.sample() for _ in range(n)]

@dataclass(frozen=True)  # frozen=True means cannot modify state (immutability)
class Die(Distribution):  # distribution for rolling n-sided die (dataclass implementation)
"""  # commented out functionality that our dataclass wrapper replaces
    def __init__(self, sides):
        self.sides = sides
    
    def __repr__(self):
        return f"Die(sides={self.sides})"
    
    def __eq__(self, other):
        if isinstance(other, Die):
            return self.sides == other.sides
        return False
"""
    sides: int  # static typing required
    
    def sample(self) -> int:
        return random.randint(1, self.sides)

@dataclass
class Gaussian(Distribution[float]):
    μ: float
    σ: float
    
    def sample(self) -> float:
        return np.random.normal(loc=self.μ, scale=self.σ)
        
    def sample_n(self, n: int) -> Sequence[float]:  # override sample_n with optimized numpy method
        return np.random.normal(loc=self.μ, scale=self.σ, size=n)
