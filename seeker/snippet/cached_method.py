#date: 2024-08-14T18:14:50Z
#url: https://api.github.com/gists/99daf49b001c0c98eb3058f5f2efdcd0
#owner: https://api.github.com/users/ktbarrett

class cached_method:

    def __init__(self, method):
        self._method = method
        update_wrapper(self, method)

    def __get__(self, instance, objtype=None):
        if instance is None:
            return self

        cache = {}

        @wraps(self._method)
        def lookup(*args, **kwargs):
            key = (args, tuple(kwargs.items()))
            try:
                return cache[key]
            except KeyError:
                res = self._method(instance, *args, **kwargs)
                cache[key] = res
                return res

        setattr(instance, self._method.__name__, lookup)
        return lookup