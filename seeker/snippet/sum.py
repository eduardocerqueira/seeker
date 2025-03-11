#date: 2025-03-11T16:53:51Z
#url: https://api.github.com/gists/00f38c59e5c37dd4bb2c3292a7ba84cb
#owner: https://api.github.com/users/mockqv

def sum(list):
    if not list:
        return 0
    return list[0] + sum(list[1:])