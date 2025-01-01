#date: 2025-01-01T16:34:17Z
#url: https://api.github.com/gists/277e7f0e372db469c507f2329d83df0b
#owner: https://api.github.com/users/MtkN1

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def from_mapping[T: DataclassInstance](
    mapping: Mapping[str, object], cls: type[T]
) -> T:
    return cls(
        **{
            field.name: mapping[field.metadata.get(from_mapping, field.name)]
            for field in fields(cls)
        }
    )


def to_mapping(obj: DataclassInstance) -> Mapping[str, object]:
    return {
        field.metadata.get(from_mapping, field.name): getattr(obj, field.name)
        for field in fields(obj)
    }


@dataclass(kw_only=True)
class Transaction:
    from_: str = field(metadata={from_mapping: "from"})
    to: str = field()
    value: int = field()


def main() -> None:
    json_literal = '{"from":"0xA1E4380A3B1f749673E270229993eE55F35663b4","to":"0x5DF9B87991262F6BA471F09758CDE1c0FC1De734","value":31337}'
    json_obj = json.loads(json_literal)

    tx = from_mapping(json_obj, Transaction)
    print(tx)

    mapping = to_mapping(tx)
    print(mapping)


if __name__ == "__main__":
    main()
