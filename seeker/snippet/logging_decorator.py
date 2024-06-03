#date: 2024-06-03T16:37:53Z
#url: https://api.github.com/gists/70dc443e6d884ceb2342f0e9f6744094
#owner: https://api.github.com/users/hoidn

import functools

def g(h):
    class Invocations:
        def __init__(self):
            self.count = 0

    invocations = Invocations()

    @functools.wraps(h)
    def wrapper(f):
        @functools.wraps(f)
        def inner(*args, **kwargs):
            invocations.count += 1
            if invocations.count <= 2:
                return h(f)(*args, **kwargs)
            else:
                return f(*args, **kwargs)

        return inner

    return wrapper

@g
def debug(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        return func(*args, **kwargs)
    return wrapper

@debug
def hello(name):
    return f"Hello, {name}!"

# First invocation
print(hello("Alice"))  # Output: Calling greet with args: ('Alice',), kwargs: {} \n Hello, Alice!

# Second invocation
print(hello("Bob"))    # Output: Calling greet with args: ('Bob',), kwargs: {} \n Hello, Bob!

# Third invocation
print(hello("Charlie"))  # Output: Hello, Charlie!