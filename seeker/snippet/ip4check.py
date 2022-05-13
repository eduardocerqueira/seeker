#date: 2022-05-13T17:18:16Z
#url: https://api.github.com/gists/dedad853345e192146766e35c569b58b
#owner: https://api.github.com/users/jamesacampbell

import re


class IsIPv4Address:
    # noinspection PyPep8Naming
    @classmethod
    def is_IPv4_address(cls, input_str: str) -> bool:
        pattern = r"\b((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}\b"
        if re.fullmatch(pattern, input_str):
            return True
        return False