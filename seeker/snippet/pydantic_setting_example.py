#date: 2024-12-23T17:08:11Z
#url: https://api.github.com/gists/865af9dfef32922b540d967c107b577e
#owner: https://api.github.com/users/madagra

from enum import StrEnum
from typing import Any, Optional
from typing_extensions import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseEnvConfig(BaseSettings):
    ciao: Optional[str] = Field(default=None, validation_alias="CIAO")
    bello: Optional[str] = Field(default=None, validation_alias="BELLO")

    @model_validator(mode='after')
    def check_is_not_none(self) -> Self:
        res = [getattr(self, field_name) is None for field_name in self.model_fields.keys()]
        print(res)
        if any(res):
            raise ValueError(f"You must set all the configuration values: "
                             f"{[item.validation_alias for item in self.model_fields.values()]}")

        return self

class ChildConfig(BaseEnvConfig):
    mio: str | None = Field(default=None, validation_alias="MIO")


if __name__ == "__main__":
    config = BaseEnvConfig()
    config2 = ChildConfig()