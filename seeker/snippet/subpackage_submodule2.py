#date: 2023-11-10T16:49:09Z
#url: https://api.github.com/gists/e882955f8046d6c35821ba53d2134bdb
#owner: https://api.github.com/users/ffissore

from myapp.mypackage.myenum import MyEnum
from . import submodule1


def call_fun():
    submodule1.is_equal(MyEnum.VALUE)
