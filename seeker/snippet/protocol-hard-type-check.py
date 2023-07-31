#date: 2023-07-31T17:05:19Z
#url: https://api.github.com/gists/e78ba0443a96fb0b1dde52a00f011f7e
#owner: https://api.github.com/users/mnogom

from abc import abstractmethod
from typing import Protocol, Callable, Any, runtime_checkable


@runtime_checkable
class LoggerProtocol(Protocol):
    @abstractmethod
    def echo(self, msg: str) -> None:
        ...

    @abstractmethod
    def special_echo(self, msg: str) -> None:
        ...


class Logger:
    def __init__(self, source: Callable[[Any], None]) -> None:
        self.__source = source

    def echo(self, msg: str) -> None:
        self.__source(msg)

    def special_echo(self, msg: str) -> None:
        self.__source(f"kf --> {msg}")


def hard_check_protocol(arg_name):
    def wrapper(fn):
        def inner(*args, **kwargs):
            expected_type = fn.__annotations__.get(arg_name)
            arg = kwargs.get(arg_name, None)
            if arg is None:
                arg_index = list(fn.__annotations__.keys()).index(arg_name)
                arg = args[arg_index]
            assert isinstance(arg, expected_type)

            return fn(*args, **kwargs)
        return inner
    return wrapper


def get_print_logger() -> Logger:
    return Logger(source=print)


@hard_check_protocol("logger")
def log(logger: LoggerProtocol, special: bool) -> None:
    if special:
        logger.special_echo("Hello")
    else:
        logger.echo("Hello")


if __name__ == "__main__":
    log(get_print_logger(), True)
    log(logger=get_print_logger(), special=False)
