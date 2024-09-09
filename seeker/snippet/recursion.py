#date: 2024-09-09T16:47:53Z
#url: https://api.github.com/gists/3fd2318829e69a34d78c503b8daf2027
#owner: https://api.github.com/users/docsallover

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)