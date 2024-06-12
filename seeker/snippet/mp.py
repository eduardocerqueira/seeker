#date: 2024-06-12T16:54:15Z
#url: https://api.github.com/gists/ad4930f46550d2b1a18b1156fda1cbaf
#owner: https://api.github.com/users/crosstyan

from typing import Any, Awaitable, Final, Generic, Optional, TypeVar, Union, cast

import anyio
import multiprocess as mp
from loguru import logger
from multiprocess.context import BaseContext, DefaultContext, assert_spawning
from multiprocess.managers import BaseManager, SyncManager
from multiprocess.pool import ApplyResult, Pool
from multiprocess.process import BaseProcess as Process
from multiprocess.queues import Empty, Full, Queue

T = TypeVar("T")


class QueueProxy(Generic[T]):
    """
    An anyio type-safe queue for multiprocessing.

    This class provides an asynchronous interface to a multiprocessing Queue,
    allowing it to be used safely with anyio without blocking the event loop.

    Note
    ------
    using the default implementation of get_state and set_state
    """

    _q: Queue
    _ctx: BaseContext

    def __init__(self, queue: Queue, ctx: BaseContext):
        """
        Initialize the QueueProxy.

        :param queue: The multiprocessing Queue to wrap.
        :param ctx: The multiprocessing context used to create the Queue.
        """
        self._q = queue
        self._ctx = ctx

    @staticmethod
    def from_manager(manager: BaseManager, size: int = 0) -> "QueueProxy[T]":
        """
        Create a new QueueProxy from a multiprocessing Manager.

        :param manager: The Manager to create the Queue from.
        :param size: The maximum size of the Queue (default 0 for unlimited).
        :return: A new QueueProxy instance.
        """
        ctx = manager._ctx  # pylint: disable=protected-access
        return QueueProxy[T](manager.Queue(size), ctx=ctx)

    def put(self, item: T, block: bool = True, timeout: Optional[float] = None):
        """
        Put an item into the queue, synchronously.

        This method wraps Queue.put() and has the same behavior.
        """
        self._q.put(item, block=block, timeout=timeout)

    async def async_put(self, item: T):
        """
        Put an item into the queue asynchronously.

        If the queue is full, this method waits until a free slot is available
        before adding the item, without blocking the event loop.
        """
        while True:
            try:
                return self._q.put_nowait(item)
            except Full:
                await anyio.sleep(0)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> T:
        """
        Get an item from the queue, synchronously.

        This method wraps Queue.get() and has the same behavior.
        """
        return cast(T, self._q.get(block=block, timeout=timeout))

    async def async_get(self):
        """
        Get an item from the queue asynchronously.

        If the queue is empty, this method waits until an item is available
        without blocking the event loop.
        """
        while True:
            try:
                return cast(T, self._q.get_nowait())
            except Empty:
                await anyio.sleep(0)

    def put_nowait(self, item: T):
        """Put an item into the queue if a free slot is immediately available."""
        self._q.put_nowait(item)

    def get_nowait(self) -> T:
        """Get an item from the queue if one is immediately available."""
        return cast(T, self._q.get_nowait())

    @property
    def queue(self) -> Queue:
        """
        Get the underlying multiprocessing Queue.

        Use this property when passing the Queue to a multiprocessing Process.
        """
        return self._q

    async def __aiter__(self):
        """
        Asynchronous iterator interface to get items from the queue.

        This allows using the queue with `async for` without blocking the event loop.
        """
        while True:
            try:
                el = self._q.get(block=False)
                yield cast(T, el)
            except Empty:
                # https://superfastpython.com/what-is-asyncio-sleep-zero/
                # yield control to the event loop
                await anyio.sleep(0)

    async def __anext__(self):
        return await self.async_get()


async def await_result(result: ApplyResult) -> Any:
    """
    wrap an ApplyResult to an awaitable
    """
    while not result.ready():
        await anyio.sleep(0)
    return result.get()


_a: Optional[int] = None
"""
this variable is expected to be unique in each process
"""


def init():
    global _a
    assert _a is None, "inited"
    _a = 1
    logger.info("init {}", _a)


def task(i: int, q: QueueProxy[int]):
    global _a
    assert _a is not None
    _a += i
    q.put(i, timeout=3)
    return _a


TIMEOUT = 5
QUEUE_SIZE = 24


def main():
    count: int = mp.cpu_count()
    logger.info("cpu count is {}", count)
    man = SyncManager()
    man.start()
    q = QueueProxy[int].from_manager(man, QUEUE_SIZE)
    p = Pool(processes=count, initializer=init)
    for _ in range(1_000):
        # https://superfastpython.com/multiprocessing-pool-asyncresult/
        ar = p.apply_async(task, (1, q))

    async def consumer():
        await q.async_put(10)
        first = await q.async_get()
        logger.info("first={}", first)
        acc = 0
        with anyio.move_on_after(TIMEOUT) as cancel_scope:
            async for i in q:
                acc += i
                cancel_scope.deadline = anyio.current_time() + TIMEOUT
        logger.info("acc={}", acc)

    anyio.run(consumer)
    p.close()


if __name__ == "__main__":
    main()
