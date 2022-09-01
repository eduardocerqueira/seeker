#date: 2022-09-01T17:19:34Z
#url: https://api.github.com/gists/23025e18eb90e03e1e2706d30c0216e9
#owner: https://api.github.com/users/m-hu

# this will not make python3 interpreter
# complain

from abc import ABCMeta, abstractmethod

class A:
    __metaclass__ = ABCMeta

    @abstractmethod
    def hello(self):
        pass

class B(A):

    def hell(self):
        print("hello")

b=B()
b.hell()