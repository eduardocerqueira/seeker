#date: 2023-09-12T17:07:33Z
#url: https://api.github.com/gists/75b1e0a017a4f1fc8deb819acc0800c2
#owner: https://api.github.com/users/mypy-play

from typing import Literal

class DataFrame:
    ...
    
@overload
def extract_s3(bucket: str, key: str, folder: Literal[False]) -> DataFrame | None:
    ...


@overload
def extract_s3(bucket: str, key: str, folder: Literal[True] = True) -> tuple[list[DataFrame], list[str]]:
    ...


def extract_s3(
    bucket: str, key: str, folder: bool = False
) -> DataFrame | None | tuple[list[DataFrame], list[str]]: