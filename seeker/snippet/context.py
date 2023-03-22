#date: 2023-03-22T17:01:11Z
#url: https://api.github.com/gists/31a8df58afdcab4939972fddc770c5ee
#owner: https://api.github.com/users/ajhebert

from contextvars import ContextVar
from pydantic import BaseSettings, Field


def hello(subj: str):
    """
    >>> hello("docstring")
    'Hello, docstring!'
    """
    return "Hello, {}!".format(subj)


# this ContextVar() is used by our test
my_var = ContextVar("my_var")


class MySettings(BaseSettings):
    var: str = Field(default_factory=my_var.get, env="test_my_var")
