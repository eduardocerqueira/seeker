#date: 2022-12-13T17:07:28Z
#url: https://api.github.com/gists/aa91f12dd855a000d1e46a4de89b418a
#owner: https://api.github.com/users/milov52

def timeit(func):
    def wrapper():
        start = datetime.now()
        result = func()
        print(datetime.now() - start)
        return result
    return wrapper