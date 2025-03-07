#date: 2025-03-07T16:42:27Z
#url: https://api.github.com/gists/5a55b532a3b8687c0d58298e5b7895d9
#owner: https://api.github.com/users/mypy-play

class A:
    def m(self, foo: int, bar: str):
        pass
    
    
class B(A):
    def m(self, foo: int, *args, **kwargs):
        pass
    
    