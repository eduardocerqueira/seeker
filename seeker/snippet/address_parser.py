#date: 2022-05-25T17:20:01Z
#url: https://api.github.com/gists/4265e285716c49c4c9f155ad696b8dc3
#owner: https://api.github.com/users/tz4678

import re
import typing as t
from dataclasses import dataclass


@dataclass
class Component:
    match: t.Match[str]
    pat: t.ClassVar[t.Pattern]

    @property
    def raw(self) -> str:
        return self.match.group()

    @property
    def identity(self) -> str:
        return self.match.group(1).lower()

    @property
    def name(self) -> str:
        return self.match.group(2).title()

    def __str__(self) -> str:
        return " ".join([self.identity, self.name])

    @classmethod
    def parse(cls, s: str) -> "Component":
        if match := cls.pat.fullmatch(s):
            return cls(match)


class City(Component):
    pat = re.compile(
        r"(г)[.\S]*\s+(\S+)",
        re.I,
    )


class Street(Component):
    pat = re.compile(
        r"(ул|мкр|пр)[.\S]*\s+(\S+)",
        re.I,
    )


class House(Component):
    pat = re.compile(
        r"(д|стр)[.\S]*\s+(\S+)",
        re.I,
    )


class Entrance(Component):
    pat = re.compile(
        r"(п|кор)[.\S]*\s+(\d+[а-я]*)",
        re.I,
    )


class Apartment(Component):
    pat = re.compile(
        r"(кв|оф|ком)[.\S]*\s+(\d+)",
        re.I,
    )


@dataclass
class Address:
    component_map: dict[t.Type[Component], Component]
    COMPONENTS: t.ClassVar[tuple[Component]] = (
        City,
        Street,
        House,
        Entrance,
        Apartment,
    )

    def __str__(self) -> str:
        return ", ".join(
            f"{self.component_map.get(c, '')}" for c in self.COMPONENTS
        ).replace(" ,", ",")

    @classmethod
    def parse(cls, address: str) -> "Address":
        parts = address.split(",")
        component_map = {}
        while parts:
            part = parts.pop()
            for klass in cls.COMPONENTS:
                if comp := klass.parse(part.strip()):
                    component_map[klass] = comp
                    continue
        return cls(component_map)


if __name__ == "__main__":
    address = Address.parse("Улица Кадырова, д-м 95, кв-ра 228, гор. Москва")
    print(str(address))  # г Москва, ул Кадырова, д 95,, кв 228
