#date: 2022-03-15T16:53:35Z
#url: https://api.github.com/gists/74f8bd4168f3653bb71a2bcaa03b0846
#owner: https://api.github.com/users/mypy-play

from typing import NewType

Gen = NewType("Gen", object)

a = Gen(1)
b = Gen("str")

reveal_locals()