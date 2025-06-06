#date: 2025-06-06T17:11:01Z
#url: https://api.github.com/gists/01afe794e3fa3f4907b5749211187633
#owner: https://api.github.com/users/mypy-play

from typing import NewType, Any

Test = NewType("Test", str)

my_test_dict: dict[Test, Any] = {}

str_idx: str = "me"
if str_idx in my_test_dict:
    val = my_test_dict[str_idx]

int_idx: int = True
my_bool_dict: dict[bool, Any] = {}
if int_idx in my_bool_dict:
    val = my_bool_dict[int_idx]
