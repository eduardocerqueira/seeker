#date: 2022-08-29T17:06:55Z
#url: https://api.github.com/gists/87f3b5796061239b828b2342aff746df
#owner: https://api.github.com/users/dot1mav

from threading import Lock, Event
from functools import wraps
from copy import deepcopy
from typing import Callable, Optional, List, Any


class Variable:
    _data: Any
    _lock: Lock
    _events: List[Event]

    def __init__(self, variable: Any) -> None:
        self._lock = Lock()
        self._events = []

        self._data = variable

    @property
    def data(self) -> Any:
        return deepcopy(self._data)

    def __is_lock(self, event: Optional[Event] = None) -> None:
        if self._lock.locked() or self._events:

            if event is None:
                event = Event()

            self._events.append(event)
            event.wait()

    def __free_event(self) -> None:
        if self._events:
            event = self._events.pop(0)
            event.set()

    def __call__(self, func: Callable[[Any], Any]) -> Callable[[Any], Any]:
        @wraps(func)
        def wraps_(*args, **kwargs):
            self.__is_lock()

            with self._lock:
                wraps_.__setattr__("data", self._data)

                result: Any = func(*args, **kwargs)

                wraps_.__delattr__("data")

            self.__free_event()

            return result

        return wraps_

    def __enter__(self) -> Any:
        self.__is_lock()

        self._lock.acquire()
        return self._data

    def __exit__(self, *exc) -> None:
        self._lock.release()
        self.__free_event()

    def __str__(self) -> str:
        return str(self._data)

    def __repr__(self) -> str:
        return f"<Variable, data = {self._data}, is_use = {self._lock.locked()}, number_waiting = {len(self._events)} >"
