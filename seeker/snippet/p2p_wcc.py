#date: 2022-06-27T16:55:36Z
#url: https://api.github.com/gists/a47ffd7b5914d8d90db1ceb0cec1ea99
#owner: https://api.github.com/users/tomasonjo

wcc = gds.wcc.mutate(G, mutateProperty="wcc")

print(wcc["componentCount"])
# 7743
print(wcc["componentDistribution"])
# {'p99': 12, 'min': 1, 'max': 11311, 'mean': 4.3564509879891515, 'p90': 4, 'p50': 2, 'p999': 39, 'p95': 6, 'p75': 3}