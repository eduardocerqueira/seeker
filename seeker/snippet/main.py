#date: 2021-12-28T17:15:00Z
#url: https://api.github.com/gists/ca4a516032b0aa07f8a9626ea3aaf77b
#owner: https://api.github.com/users/mypy-play

from typing import Callable, Concatenate, ParamSpec, TypeVar

P = ParamSpec('P')
T = TypeVar('T')

def pass_context(f: Callable[Concatenate[Context, P], T]) -> Callable[P, T]:
    """Marks a callback as wanting to receive the current context
    object as first argument.
    """

    def wrapper(*args: P.args, **kwargs: P.kwargs):
        return f(get_current_context(), *args, **kwargs)

    return wrapper


# dummy definitions so mypy finds something for "get_current_context" and "Context"

class Context: pass
def get_current_context() -> Context: return Context()