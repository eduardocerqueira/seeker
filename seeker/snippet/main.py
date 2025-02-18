#date: 2025-02-18T16:48:47Z
#url: https://api.github.com/gists/9d84be8c0f72a283e042049e5c4d51f9
#owner: https://api.github.com/users/mypy-play

from random import choice

user = choice([None, "something"])
if user is None or len(user) > 8:
    print("Username too long", len(user))
