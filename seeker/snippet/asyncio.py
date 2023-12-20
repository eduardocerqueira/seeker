#date: 2023-12-20T17:04:21Z
#url: https://api.github.com/gists/8cdf241b6233855543fe3b3415525edb
#owner: https://api.github.com/users/elpekenin

import asyncio
from abc import abstractmethod, ABC


class Task(ABC):
    @abstractmethod
    async def loop(self):
        ...


class TaskA(Task):
    async def loop(self):
        while True:
            await asyncio.sleep(1)
            print("A")


class TaskB(Task):
    async def loop(self):
        while True:
            await asyncio.sleep(2)
            print("B")


class TaskC(Task):
    """This provides some shared logic for children to use"""
    def very_cool_func(self):
        print(42)


class TaskD(TaskC):
    async def loop(self):
        while True:
            await asyncio.sleep(3)
            print("D")
            self.very_cool_func()

def _fetch_subclasses_impl(klass: type, subclasses: set[type]):
    """Recursively calls itself, to fetch all classes."""

    # skip abstract classes (ie: don't implement loop)
    if len(klass.__abstractmethods__) == 0:
        subclasses.add(klass)

    # iterate subclasses (if any)
    for subclass in klass.__subclasses__():
        _fetch_subclasses_impl(subclass, subclasses)


def fetch_subclasses(klass: type) -> set[type]:
    """Convenience wrapper around the recursive logic."""
    subclasses = set()
    _fetch_subclasses_impl(klass, subclasses)
    return subclasses


class Command(BaseCommand):
    async def run_workers(self):
        tasks = [
            asyncio.Task(subclass().loop())
            for subclass in fetch_subclasses(Task)
        ]

        await asyncio.gather(*tasks)

    def handle(self, *args, **options):
        asyncio.run(self.run_workers())
