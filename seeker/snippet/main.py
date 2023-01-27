#date: 2023-01-27T16:48:08Z
#url: https://api.github.com/gists/4782e194ffd274c0e80aee1b747d6594
#owner: https://api.github.com/users/mypy-play

# This works.

class A:
    def example(self) -> int:
        return 1
    
class B(A):
    def example(self) -> bool:
        return True


# This does not.

class C:
    def example(self) -> list[int]:
        return [1]
    
class D(C):
    def example(self) -> list[bool]:
        return [True]