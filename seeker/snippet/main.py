#date: 2025-11-24T17:04:49Z
#url: https://api.github.com/gists/d4046bee1113d1293d76142d0830843a
#owner: https://api.github.com/users/mypy-play

from typing import Any, Callable, cast, reveal_type, Concatenate


# Example for https://github.com/python/cpython/pull/121693

def copy_func_params[**Param, RV](
    source_func: Callable[Param, Any]
) -> Callable[[Callable[..., RV]], Callable[Param, RV]]:
    """Cast the decorated function's call signature to the source_func's.

    Use this decorator enhancing an upstream function while keeping its
    call signature.
    Returns the original function with the source_func's call signature.

    Usage::

        from typing import copy_func_params, Any

        def upstream_func(a: int, b: float, *, double: bool = False) -> float:
            ...

        @copy_func_params(upstream_func)
        def enhanced(
            a: int, b: float, *args: Any, double: bool = False, **kwargs: Any
        ) -> str:
            ...

    .. note::

       Include ``*args`` and ``**kwargs`` in the signature of the decorated
       function in order to avoid TypeErrors when the call signature of
       *source_func* changes.
    """

    def return_func(func: Callable[..., RV]) -> Callable[Param, RV]:
        return cast(Callable[Param, RV], func)

    return return_func
    
    


def upstream_func(a: int, b: float, *, double: bool = False) -> float:
    return 1.0

@copy_func_params(upstream_func)
def enhanced(
    *args: Any, double: bool = False, **kwargs: Any
) -> str:
    return ""
    
number: int


# Expected two typing errors, on for the return type, one for a
number = enhanced(a="1", b=True)


def copy_method_params[**Param, Arg1, RV](
    source_method: Callable[Concatenate[Any, Param], Any]
) -> Callable[
    [Callable[Concatenate[Arg1, ...], RV]],
    Callable[Concatenate[Arg1, Param], RV]
]:
    """Cast the decorated method's call signature to the source_method's.

    Same as :func:`copy_func_params` but intended to be used with methods.
    It keeps the first argument (`self`/`cls`) of the decorated method.
    """

    def return_func(
        func: Callable[Concatenate[Arg1, ...], RV]
    ) -> Callable[Concatenate[Arg1, Param], RV]:
        return cast(Callable[Concatenate[Arg1, Param], RV], func)

    return return_func
    
class A:
    def __init__(self, val: int) -> None:
        self.val = val
        
class B(A): ...


class Orig:
    def method(self, a: int, b: float, *, double: bool = False) -> A:
        return A(int(a*b))
        

class Composition:
    def __init__(self):
        self.orig = Orig()

    @copy_method_params(Orig.method)
    def method(self, a: int|str, b: float|str, **kwargs: Any) -> B:
        result = self.orig.method(int(a), float(b), **kwargs).val
        return B(result)
        
    # example why `copy_func_params` does not work for methods
    # at least when using composition.
    # for Inheritance this could work partly
    @copy_func_params(Orig.method)
    def fail(self, a: int|str, b: float|str, **kwargs: Any) -> B:
        result = self.orig.method(int(a), float(b), **kwargs).val
        return B(result)
        
    def example(self) -> tuple[B, B]:
        # fail has wrong self argument
        return self.method(1,1), self.fail(1,1)
        

reveal_type(Orig.method)
reveal_type(Composition.method)
# This shows the wrong type for self
reveal_type(Composition.fail)
reveal_type(Orig().method)
reveal_type(Composition().method)
# can't call fail because has wrong types
reveal_type(Composition().fail)

b: B

# for first argument there should be a type error
# but it should be possible to assign the value
b = Composition().method("1", 2)

# invalid self argument
b = Composition().fail("1", 2)
