#date: 2025-06-06T16:48:52Z
#url: https://api.github.com/gists/cc48b01ee289cd914f48b5405d2293bd
#owner: https://api.github.com/users/RampantLions

_swizzle_backup = {}

def swizzle(cls, method_name, wrapper_func):
    original = getattr(cls, method_name)
    _swizzle_backup[(cls, method_name)] = original
    setattr(cls, method_name, wrapper_func(original))

def unswizzle(cls, method_name):
    key = (cls, method_name)
    if key in _swizzle_backup:
        setattr(cls, method_name, _swizzle_backup.pop(key))

# Usage
class Target:
    def greet(self):
        return "Hi"

def wrapper(orig):
    def new(self):
        return f"{orig(self)} World"
    return new

swizzle(Target, 'greet', wrapper)
print(Target().greet())  # Hi World
unswizzle(Target, 'greet')
print(Target().greet())  # Hi
