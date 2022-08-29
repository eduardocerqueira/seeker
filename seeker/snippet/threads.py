#date: 2022-08-29T17:05:04Z
#url: https://api.github.com/gists/11eab0c4f81045ca158f52c6dab8227e
#owner: https://api.github.com/users/dot1mav

from abc import ABC, abstractmethod
from threading import Event, Thread, ThreadError
from copy import deepcopy
from typing import Callable, Optional, Any, Iterable, Dict


class BaseThread(ABC):
    _target: Callable[[Any], Any]
    _args: Iterable[Any]
    _name: str
    _more_option: Dict[str, Any]
    _thread: Optional[Thread]

    def __init__(self, target: Callable[[Any], Any], args: Optional[Iterable[Any]] = None, name: Optional[str] = None,
                 **kwargs) -> None:

        self._target = target

        self._args = args if args is not None else ()
        self._name = name if name is not None else self._target.__name__

        self._more_option = kwargs

        self._thread = None

    @property
    def options(self) -> Dict[str, Any]:
        return deepcopy(self._more_option)

    @property
    def function(self) -> Callable[[Any], Any]:
        return self._target

    @property
    def arguments(self) -> Iterable[Any]:
        return deepcopy(self._args)

    @arguments.setter
    def arguments(self, value: Iterable[Any]) -> None:
        if not (isinstance(value, tuple) or isinstance(value, list)):
            raise TypeError("arguments must be iterable")
        self._args = value

    @abstractmethod
    def start(self, new_args: Optional[Iterable[Any]] = None) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    @abstractmethod
    def is_stop(self) -> bool:
        ...

    def __setitem__(self, key: str, value: Any) -> None:
        if not isinstance(key, str):
            raise TypeError("Key must be string")
        self._more_option.update({key: value})

    def __getitem__(self, key: str) -> Optional[Any]:
        if not isinstance(key, str):
            raise TypeError("key must be string")
        return self._more_option[key] if key in self._more_option else None

    def __delitem__(self, key: str) -> None:
        if key in self._more_option:
            return self._more_option.__delitem__(key)

    def __repr__(self) -> str:
        return f"<name={self._name}, function={self._target}, args={self._args}>"


class StoppableThread(BaseThread):

    def start(self, *args) -> None:
        if self._thread is not None:
            raise RuntimeError("Thread still running")
        if args:
            self.arguments = args

        self._thread = Thread(target=self.function, args=self.arguments, name=self._name, **self._more_option)
        self._thread.__setattr__("is_stop", False)

        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            raise ThreadError("Thread doesn't exist")
        self._thread.is_stop = True
        self._thread = None

    def is_stop(self) -> bool:
        return self._thread is None


class IntervalThread(BaseThread):
    _interval: float
    _event: Event

    def __init__(self, target: Callable[[Any], Any], args: Optional[Iterable[Any]] = None,
                 name: Optional[str] = None, interval: Optional[float] = None) -> None:
        super().__init__(target, args, name)

        self._interval = interval if interval is not None else 30
        self._event = Event()
    
    def _run(self) -> None:
        while not self._event.is_set():
            self.function(*self.arguments)
            self._event.wait(self._interval)
        
        self._event.clear()

    def start(self, *, new_args: Optional[Iterable[Any]] = None, new_interval: Optional[float] = None) -> None:
        if self._thread is not None:
            raise RuntimeError("Thread still running")

        if new_args is not None:
            self.arguments = new_args

        if new_interval is not None:
            self._interval = new_interval

        self._thread = Thread(target=self._run, name=self._name)

        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            raise ValueError("Thread doesn't exist")

        self._event.set()

        self._thread = None

    def is_stop(self) -> bool:
        return self._thread is None
