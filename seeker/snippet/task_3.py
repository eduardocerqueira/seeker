#date: 2025-08-13T16:40:41Z
#url: https://api.github.com/gists/c16e305a437ee7787571bd8dbb39ee8c
#owner: https://api.github.com/users/DanielIvanov19

import inspect


def accepts(expected_count):
    def decorator(func):
        # Get function signature
        sig = inspect.signature(func)
        actual_count = len(sig.parameters)

        if actual_count != expected_count:
            raise TypeError(
                f"{func.__name__} should have {expected_count} arguments, "
                f"but has {actual_count}"
            )

        return func

    return decorator