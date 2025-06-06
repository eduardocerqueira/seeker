#date: 2025-06-06T16:48:52Z
#url: https://api.github.com/gists/cc48b01ee289cd914f48b5405d2293bd
#owner: https://api.github.com/users/RampantLions

def swizzle_method(cls, method_name):
    old_method = getattr(cls, method_name)

    def new_method(self, *args, **kwargs):
        print(f"{method_name} called with args: {args}")
        return old_method(self, *args, **kwargs)

    setattr(cls, method_name, new_method)

class Greeter:
    def greet(self, name):
        return f"Hello, {name}"

swizzle_method(Greeter, "greet")

print(Greeter().greet("Josh"))  # Logs call, then "Hello, Josh"
