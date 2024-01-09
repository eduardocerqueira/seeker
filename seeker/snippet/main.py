#date: 2024-01-09T16:51:35Z
#url: https://api.github.com/gists/3166a2e3049ca2655ebca146a6eb4173
#owner: https://api.github.com/users/mypy-play

a = [('a', 1), ('b', 2), ('c', 3)]
reveal_type(tuple(zip(*a)))


b: list[tuple[float, str, bytes]] = []

reveal_type(zip(*b))
