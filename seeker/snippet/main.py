#date: 2022-05-27T17:09:41Z
#url: https://api.github.com/gists/cd678ce469815849a9ac8b10ffcd24a6
#owner: https://api.github.com/users/mypy-play

from typing import overload, TypeVar, Callable, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


@overload
def decorator(fn: Callable[P, T]) -> Callable[P, T]:
    ...


@overload
def decorator(*args: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    ...


def decorator(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        print("without argument")
        fn = args[0]

        def inner(*args, **kwargs):
            print("start")
            result = fn(*args, **kwargs)
            print("finish")
            return result

        return inner
    else:
        print("with argument")

        def real_decorator(fn):
            def inner(*args, **kwargs):
                print("start")
                result = fn(*args, **kwargs)
                print("finish")
                return result

            return inner

        return real_decorator


@decorator
def fct0(a: int, b: int) -> int:
    return a * b


@decorator("foo", "bar")  # any number of arguments
def fct1(a: int, b: int) -> int:
    return a * b


print(fct0(10, 20))
print(fct1(30, 40))