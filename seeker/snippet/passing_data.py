#date: 2023-07-05T16:41:20Z
#url: https://api.github.com/gists/a771db3570cdd50babfcb781bc63caf4
#owner: https://api.github.com/users/schwehr

#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Google Inc. All Rights Reserved.

# I recommend that dataclass decorator option.

import collections
import dataclasses

a_tuple_of_lists = (['a', 'b', 'c'], ['d', 'e'])
print('a_tuple_of_lists', a_tuple_of_lists, type(a_tuple_of_lists))
print()

a_tuple_of_tuples = (('a', 'b', 'c'), ('d', 'e'))
print('a_tuple_of_tuples', a_tuple_of_tuples, type(a_tuple_of_tuples))
print()

MyNamedTuple = collections.namedtuple('MyType', 'Thing1 Thing2')
a_named_tuple =  MyNamedTuple(['a', 'b', 'c'], ['d', 'e'])
print('a_named_tuple', a_named_tuple, type(a_named_tuple))
print('  Thing1:', a_named_tuple.Thing1)
print()

a_dict = {'thing1': ['a', 'b', 'c'], 'thing2': ['d', 'e']}
print('a_dict', a_dict, type(a_dict))
print()


@dataclasses.dataclass(frozen=True)
class MyType:
    thing1: tuple[tuple[str]]
    thing2: tuple[tuple[str]]

my_things = MyType(('a', 'b'),  ('c', 'd', 'e'))
print('my_things', my_things, type(my_things))
print('  thing2:', my_things.thing2)
print()

