#date: 2022-09-09T17:12:23Z
#url: https://api.github.com/gists/89bf693e2f3a6b2c8df0a405c334f0cf
#owner: https://api.github.com/users/jellyCodee

from enum import Enum


class SourceEnum(str, Enum):
    English = "English"
    French = "French"
    Romanian = "Romanian"
    German = "German"


class DestinationEnum(str, Enum):
    English = "English"
    French = "French"
    Romanian = "Romanian"
    German = "German"