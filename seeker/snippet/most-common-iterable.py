#date: 2022-08-01T17:19:58Z
#url: https://api.github.com/gists/a27673d9743f4d222e985392923df1a2
#owner: https://api.github.com/users/dublado

# collections.Counter lets you find the most common
# elements in an iterable:

>>> import collections
>>> c = collections.Counter('helloworld')

>>> c
Counter({'l': 3, 'o': 2, 'e': 1, 'd': 1, 'h': 1, 'r': 1, 'w': 1})

>>> c.most_common(3)
[('l', 3), ('o', 2), ('e', 1)]