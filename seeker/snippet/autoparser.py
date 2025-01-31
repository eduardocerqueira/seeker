#date: 2025-01-31T16:41:37Z
#url: https://api.github.com/gists/55f4cc650775f7bf1645abe3bf811d3e
#owner: https://api.github.com/users/markusand

from typing import List, Type, TypeVar

T = TypeVar("T", bound="Autoparser")

class Autoparser:
    """Automatic parse into parseable classes."""

    def __repr__(self):
        exclude = (classmethod,)
        attrs = [
            f"{name}={getattr(self, name)}"
            for name, field in self.__class__.__annotations__.items()
            if field not in exclude
        ]
        return f"{self.__class__.__name__}({', '.join(attrs)})"

    @classmethod
    def validate(cls: Type[T], data: str) -> None:
        """Validate Raises ValueError if invalid."""
        raise NotImplementedError("Not implemented")

    @classmethod
    def split(cls: Type[T], data: str) -> List[str]:
        """Split sentence into parts"""
        raise NotImplementedError("Not implemented")

    @classmethod
    def parse(cls: Type[T], data: str):
        """Parse sentence into a class instance"""
        cls.validate(data)
        values = cls.split(data)

        exclude = (classmethod, property)
        fields = {
            name: type_hint
            for name, type_hint in cls.__annotations__.items()
            if type_hint not in exclude
        }

        if len(values) != len(fields):
            raise ValueError(f"Expected {len(fields)} values, got {len(values)}")

        parsed = {
            name: field(value) if value else None
            for (name, field), value in zip(fields.items(), values)
        }

        return cls(**parsed)