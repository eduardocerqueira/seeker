#date: 2023-06-07T16:46:32Z
#url: https://api.github.com/gists/e00296d3c4b4daa245dd2b2fac4a6de6
#owner: https://api.github.com/users/mypy-play

import abc
import functools
import inspect
import typing

class Service(abc.ABC):
    ...


class ServiceA(Service):
    @staticmethod
    @abc.abstractmethod
    def method_a(a: int) -> str:
        ...


class ServiceA1(ServiceA):
    @staticmethod
    def method_a(a: int) -> str:
        return f"A1: {a}"


def foobar(foo: int, service_a: ServiceA) -> str:
    return service_a.method_a(foo)


def inject(func: typing.Callable, *services: typing.Type[abc.ABC]) -> typing.Callable:
    annotations = typing.get_type_hints(func)
    del annotations["return"]

    bind_services = {
        key: service
        for key, value in annotations.items()
        if issubclass(value, abc.ABC)
        for service in services
        if issubclass(service, value)
    }

    partial_func = functools.partial(func, **bind_services)

    # Update the new function's signature
    new_params = [
        param
        for name, param in inspect.signature(func).parameters.items()
        if name not in bind_services
    ]
    new_sig = inspect.Signature(new_params)
    functools.update_wrapper(partial_func, func)  # Copy attributes from the original function
    partial_func.__signature__ = new_sig

    return partial_func


foobar_A1 = inject(foobar, ServiceA1)
reveal_type(foobar_A1)