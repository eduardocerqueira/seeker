#date: 2022-06-15T16:47:20Z
#url: https://api.github.com/gists/938f24eb3e8b0e150f042dbf5cbdb299
#owner: https://api.github.com/users/mypy-play

class A:
    def __init__(self, a: int, b: str):
        self.a = a
        self.b = b
    
A(3, "4")
A.b = 4
