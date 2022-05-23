#date: 2022-05-23T17:00:52Z
#url: https://api.github.com/gists/cbae00690656346859c94bc9313b48e8
#owner: https://api.github.com/users/mypy-play

"""
Proposal for metadata:
We should have a single object of all possible metadata that either has a typed value (e.g "datetime" for
time_vested_datetime") or some "unknown" value.

Within a single transaction when you are updating a share line (e.g within a single adapter), the metadata might have
mixed types of values (e.g "datetime" as well as "unknown"), but once you are done with the transaction, all the values
must either be unknown or known. You should be able to give a default value share sections without explicit data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, Sequence, List, Union, Tuple, TypeVar, Generic, ClassVar, Type, Callable, Optional, Any

from typing_extensions import TypeGuard


#### Metadata Implementation detail ####
class UnknownMetadataType(Enum):
    UNKNOWN = "UNKNOWN"


class MissingDefaultException(Exception):
    pass


def raise_missing_default_exception():
    raise MissingDefaultException


def default_to_none():
    return None


def default_to_false():
    return False


MetadataValueType = TypeVar("MetadataValueType", bound=Any)


class MetadataFieldName(Enum):
    # Enum has value name, type of value, and a default callable
    is_blocked = "is_blocked", bool, default_to_false
    canceled_datetime = "canceled_datetime", Optional[datetime], default_to_none
    time_vesting_datetime = "time_vesting_datetime", datetime, raise_missing_default_exception

    def __init__(
        self, _: str, value_type: Type[MetadataValueType], default_value_function: Callable[[], Type[MetadataValueType]]
    ) -> None:
        self._value_type = value_type
        self._default_value_function = default_value_function

    def __str__(self) -> str:
        return self.value

    @property
    def value_type(self) -> Type[MetadataValueType]:
        return self._value_type

    @property
    def default_value(self) -> Callable[[], Type[MetadataValueType]]:
        return self._default_value_function


@dataclass(frozen=True)
class MetadataValue(Generic[MetadataValueType]):
    field_name: ClassVar[MetadataFieldName]
    value: Union[MetadataValueType, UnknownMetadataType]


@dataclass(frozen=True)
class BlockedMetadataValue(MetadataValue[Union[bool, UnknownMetadataType]]):
    field_name = MetadataFieldName.is_blocked
    value: Union[bool, UnknownMetadataType]


@dataclass(frozen=True)
class TimeVestingMetadataValue(MetadataValue[datetime]):
    field_name = MetadataFieldName.time_vesting_datetime
    value: datetime


@dataclass(frozen=True)
class CanceledMetadataValue(MetadataValue[Union[Optional[datetime], UnknownMetadataType]]):
    field_name = MetadataFieldName.time_vesting_datetime
    value: Union[Optional[datetime], UnknownMetadataType]


@dataclass(frozen=True)
class Metadata:
    # - Ensures only one field per metadata (e.g time_vested_datetime only happens once)
    # - Can add validation methods to ensure that it only contains the required metadata, no more, no less (e.g vesting
    # will have its own subset of metadata)
    # - Easy to retrieve
    metadata_values: Dict[MetadataFieldName, MetadataValue[Any]]


class KnownMetadata(Metadata):
    ...


class ValidMetadata(Metadata):
    ...


# when we have completed building a shareline, we can only pass out known metadata to the external word.
# Furthermore, each section can mandate its own required metadata - e.g the vesting adapters should make sure that when
# they are done with a shareline, all vesting data is known.
def is_known_metadata(metadata: Metadata) -> TypeGuard[KnownMetadata]:
    return all(isinstance(m.value, m.value_type) for m in metadata.metadata_values)


# When we are building the shareline, within each step, we should only pass valid metadata between each adapters
def is_valid_metadata(metadata: Metadata) -> TypeGuard[ValidMetadata]:
    return is_known_metadata(metadata) or all(
        isinstance(m.value, UnknownMetadataType) for m in metadata.metadata_values
    )


@dataclass(frozen=True)
class ShareSection:
    start_index_inclusive: Decimal
    end_index_inclusive: Decimal
    metadata: Metadata

    @classmethod
    def build(cls, start_index_inclusive: Decimal, end_index_inclusive: Decimal, metadata_values: List[MetadataValue]):
        metadata_values_in_dict = {metadata_value.field_name: metadata_value for metadata_value in metadata_values}
        return cls(
            start_index_inclusive=start_index_inclusive,
            end_index_inclusive=end_index_inclusive,
            metadata=Metadata(metadata_values=metadata_values_in_dict),
        )


class Strategy(Enum):
    """
    Describes strategy of how to apply metadata to tranches.
    """

    LIFO = "LIFO"  # last in, first out means we apply from the back
    FIFO = "FIFO"  # first in, first out means we apply from the front


@dataclass(frozen=True)
class Shareline:
    sections: Sequence[ShareSection]

    def add_metadata(
        self,
        start_index_inclusive: Decimal,
        end_index_exclusive: Decimal,
        value: MetadataValue,
    ):
        ...

    def add_metadata_to_sections_excluding(
        self,
        quantity: Decimal,
        metadata_value: MetadataValue,
        excluding_sections_with_metadata: List[MetadataValue],
        strategy: Strategy = Strategy.FIFO,
    ):
        # deterministically adds metadata to available quantity, but you can exclude shares like "canceled=True"
        ...

    def replace_unknown_metadata_fields_with_default(self, field_name: MetadataFieldName):
        ...

    def merge(self, other_shareline: Shareline):
        ...


# Creating metadata is easy and nicely typed
metadata_values_1: List[MetadataValue] = [
    TimeVestingMetadataValue(value=datetime(2022, 1, 1, 0, 0)),
    BlockedMetadataValue(value=UnknownMetadataType.UNKNOWN),
    CanceledMetadataValue(value=UnknownMetadataType.UNKNOWN),
]

metadata_values_2: List[MetadataValue] = [
    # note that we can add custom mappers to let any other value override `UNKNOWN` but any conflicting values can error
    CanceledMetadataValue(value=None),
]

# store it as a dict on the share line for easy access
section1 = ShareSection.build(
    start_index_inclusive=Decimal(0), end_index_inclusive=Decimal(200), metadata_values=metadata_values_1
)
share_line1 = Shareline(sections=[section1])

section2 = ShareSection.build(
    start_index_inclusive=Decimal(0), end_index_inclusive=Decimal(200), metadata_values=metadata_values_2
)
share_line2 = Shareline(sections=[section2])


share_line1.merge(other_shareline=share_line2)