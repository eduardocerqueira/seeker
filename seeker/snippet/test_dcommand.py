#date: 2022-06-08T17:06:40Z
#url: https://api.github.com/gists/65fa55a901c98c5545a494faa749624c
#owner: https://api.github.com/users/rec

from engora.misc import dcommand
from typer import Argument, Option, Typer
from typing import Optional
import pytest

command = Typer().command


@command(help='test')
def a_command(
    bucket: str = Argument(
        ..., help='The bucket to use'
    ),

    keys: str = Argument(
        'keys', help='The keys to download'
    ),

    pid: Optional[int] = Option(
        None, help='pid'
    ),
):
    ACommand(**locals())()


@dcommand(a_command)
class ACommand:
    def __call__(self):
        return self.bucket, self.keys, self.pid


@dcommand(a_command)
def a_function(self):
    return self.bucket, self.keys, self.pid


def test_dcommand():
    assert ACommand('bukket')() == ('bukket', 'keys', None)
    assert ACommand('bukket', 'kois', pid=3)() == ('bukket', 'kois', 3)

    match = 'missing 1 required positional argument: \'bucket\''
    with pytest.raises(TypeError, match=match):
        ACommand()
