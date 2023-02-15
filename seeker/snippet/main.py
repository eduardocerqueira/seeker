#date: 2023-02-15T16:47:50Z
#url: https://api.github.com/gists/fcebc2af2314bceecf012586ff13ea30
#owner: https://api.github.com/users/mypy-play


from typing import TypeVar, List

class A:
    def __init__(self, a):
        self.a=a
class B(A):
    def __init__(self, a, b):
        super().__init__(a)
        self.b=b

x1:B
x1=B(1,2)
print(x1.b)

x2:B
x2=A(2)