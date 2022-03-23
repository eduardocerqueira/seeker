#date: 2022-03-23T17:12:21Z
#url: https://api.github.com/gists/02889e2ae96e99467dc53802579e0846
#owner: https://api.github.com/users/mypy-play

from typing import Callable, List, Any, Dict, Type, TypeVar, Optional, Generic


class Event:
    field: Optional[str]


class EventWithField(Event):
    field: str


def foo(event: Event) -> Optional[str]:
    if isinstance(event, EventWithField):
        return foo2(event=event)


def foo2(event: EventWithField) -> str:
    return event.field
    

def solution_1(event: Event) -> str:
    if event.field:
        field = event.field
    # Possibly raise exception?
    # else:
    #     raise Exception()
    return field