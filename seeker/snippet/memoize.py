#date: 2022-01-17T16:57:20Z
#url: https://api.github.com/gists/bc5ddbc063508cbd5a6031e6595bbe41
#owner: https://api.github.com/users/qorrect

def memoize(f):
    """ Memoization decorator for a function taking one or more arguments. """
    class memodict(dict):
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)

        def __missing__(self, key):
            ret = self[key] = f(*key)
            return ret

    return memodict().__getitem__