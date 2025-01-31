#date: 2025-01-31T16:41:37Z
#url: https://api.github.com/gists/55f4cc650775f7bf1645abe3bf811d3e
#owner: https://api.github.com/users/markusand

from functools import reduce
from typing import List, Type, TypeVar
from .autoparser import Autoparser

T = TypeVar("T", bound="NMEA")

class NMEA(Autoparser):
    """NMEA class"""

    @classmethod
    def validate(cls: Type[T], data: str) -> None:
        """Validate NMEA message."""
        try:
            if not data or len(data) == 0:
                raise ValueError("Empty data")

            content, checksum = data.strip("\n").split("*", 1)

            if len(checksum) != 2:
                raise ValueError("Checksum length must be 2 digits")

            if not content.startswith("$") or cls.__name__ not in (content[3:6], "NMEA"):
                raise ValueError(f"{content[:6]} is an invalid NMEA identifier")

            # Verify checksum
            _checksum = reduce(lambda x, y: x ^ ord(y), content[1:], 0)
            if _checksum != int(checksum, 16):
                raise ValueError("Checksum verification failed")

        except ValueError as error:
            raise ValueError("Invalid or malformed NMEA message") from error

    @classmethod
    def split(cls: Type[T], data: str) -> List[str]:
        """Split sentence into NAME parts"""
        content, _checksum = data[1:].split("*")
        return content.split(",")

    @property
    def talker(self) -> str:
        """Get talker"""
        sentence = getattr(self, "sentence", "")
        return sentence[:2]