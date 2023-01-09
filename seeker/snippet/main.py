#date: 2023-01-09T16:48:52Z
#url: https://api.github.com/gists/e36bc46aea1e76c1c10c5c7e69a2ad3b
#owner: https://api.github.com/users/mypy-play

from fastapi import Request


def get_request(request: Request) -> None:
    print(request.client.host)