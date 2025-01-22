#date: 2025-01-22T17:12:53Z
#url: https://api.github.com/gists/96a9840ea1cc7597dc73fa4f0459f556
#owner: https://api.github.com/users/mypy-play

def truncate[S:(str, bytes)](s: S) -> S:
    return s[:10]
    
def log(v: str | bytes):
    assert isinstance(v, (str, bytes))
    print(truncate(v))  # error: Value of type variable "S" of "truncate" cannot be "str | bytes"  [type-var]
    
def log2(v: str | bytes):
    if isinstance(v, str):
        print(truncate(v))
    else:
        print(truncate(v))
