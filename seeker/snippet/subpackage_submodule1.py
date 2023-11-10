#date: 2023-11-10T16:49:09Z
#url: https://api.github.com/gists/e882955f8046d6c35821ba53d2134bdb
#owner: https://api.github.com/users/ffissore

from ..myenum import MyEnum


def is_equal(myenum: MyEnum):
    print(id(MyEnum.VALUE))
    print(id(myenum))
    print(myenum == MyEnum.VALUE)
