#date: 2022-12-14T17:05:04Z
#url: https://api.github.com/gists/8dac6763092cf5017a93d84e8d282b2d
#owner: https://api.github.com/users/mypy-play

from typing import List, Type, cast

class Base(object):
    pass

class ChildOne(Base):
    pass

class ChildTwo(Base):
    pass

my_list: List[Type[Base]] = [ChildOne, ChildTwo]

my_new_list: List[Type[Base]] = [ChildOne if x == ChildTwo else x for x in my_list]

my_other_list: List[Type[Base]] = [cast(Type[Base], ChildOne) if x == ChildTwo else x for x in my_list]