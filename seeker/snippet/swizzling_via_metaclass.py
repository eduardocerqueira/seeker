#date: 2025-06-06T16:48:52Z
#url: https://api.github.com/gists/cc48b01ee289cd914f48b5405d2293bd
#owner: https://api.github.com/users/RampantLions

class SwizzleMeta(type):
    def __new__(cls, name, bases, dct):
        for attr, val in dct.items():
            if callable(val) and not attr.startswith("__"):
                def wrap(fn):
                    def swizzled(self, *args, **kwargs):
                        print(f"Calling: {fn.__name__}")
                        return fn(self, *args, **kwargs)
                    return swizzled
                dct[attr] = wrap(val)
        return super().__new__(cls, name, bases, dct)

class Logger(metaclass=SwizzleMeta):
    def foo(self): return "foo"
    def bar(self): return "bar"

l = Logger()
print(l.foo())  # Logs then returns "foo"
print(l.bar())  # Logs then returns "bar"

