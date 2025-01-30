#date: 2025-01-30T16:55:49Z
#url: https://api.github.com/gists/230fb8318cda0192c27a4a9cecf92a05
#owner: https://api.github.com/users/zzstoatzz

from prefect import task


def some_fn(x: int, y: str) -> int:
    return x + len(y)


some_task = task(some_fn)

print(some_task(x=1, y="hello"))
