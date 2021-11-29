#date: 2021-11-29T17:04:07Z
#url: https://api.github.com/gists/48f2b7256e7c41027c536d45de85652d
#owner: https://api.github.com/users/pigmonchu

def g(f):
    def k(*args):
        return 2 * f(*args)
    return k
        