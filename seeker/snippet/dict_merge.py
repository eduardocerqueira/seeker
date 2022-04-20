#date: 2022-04-20T17:14:28Z
#url: https://api.github.com/gists/c70bcd434beda50b4c7feae4a41386cb
#owner: https://api.github.com/users/FerusAndBeyond

a = { "a": 5, "b": 5 }
b = { "c": 5, "d": 5 }
c = { **a, **b }
assert c == { "a": 5, "b": 5, "c": 5, "d": 5 }
# order matters!
# the last added will overwrite the first added
c = { **a, **b, "a": 10 }
assert c == { "a": 10, "b": 5, "c": 5, "d": 5 }
b["a"] = 10
c = { **a, **b }
assert c == { "a": 10, "b": 5, "c": 5, "d": 5 }
# this doesn't work when initializing with dict(...)
c = dict(**a, **b)
# => TypeError: type object got multiple values for keyword argument 'a'