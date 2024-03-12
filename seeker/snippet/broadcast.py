#date: 2024-03-12T16:42:58Z
#url: https://api.github.com/gists/2807f34905a3099137f905d107489376
#owner: https://api.github.com/users/charbonnierg

"""Usage:


def get_listener(request: Request) -> Listener:
    '''Function used to access listener from the application state.'''
    return request.app.state.listener


def set_listener(app: FastAPI, url: str) -> None:
    '''Function to be called once on application startup.'''
    app.state.listener = Listener(url)


async def example(listener: Listener = Depends(get_listener)) -> None:
    async with listener.subscribe("test") as subscription:
        async for event in subscription:
            print(event)


async def other_example(listener: Listener = Depends(get_listener)) -> None:
    async with listener.subscribe("test") as subscription:
        while True:
            event = await subscription.next()
            if event is None:
                break
            print(event)
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncContextManager

from broadcaster import Broadcast, Event
from broadcaster._base import Subscriber, Unsubscribed


class Subscription:
    """A wrapper around a subscriber to make it an async iterator."""

    def __init__(
        self,
        channel: str,
        listener: Listener,
    ) -> None:
        self.channel = channel
        self._listener = listener
        self._context: AsyncContextManager[Subscriber] | None = None
        self._subscriber: Subscriber | None = None

    def __aiter__(self) -> Subscription:
        return self

    async def __aenter__(self) -> Subscription:
        """Start the subscription as an async context manager."""
        await self._listener._connect_subscription(self)
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Stop the subscription as an async context manager."""
        try:
            if self._context:
                await self._context.__aexit__(exc_type, exc, tb)
        finally:
            await self._listener._remove_subscription(self)

    async def __anext__(self) -> Event:
        """Return next event or raise StopAsyncIteration if subscription is closed."""
        if not self._subscriber:
            raise RuntimeError("Subscription not started")
        try:
            return await self._subscriber.get()
        except Unsubscribed:
            raise StopAsyncIteration

    async def next(self) -> Event | None:
        """Return next event or None is subscription is closed."""
        if not self._subscriber:
            raise RuntimeError("Subscription not started")
        try:
            return await self._subscriber.get()
        except Unsubscribed:
            return None


class Listener:
    """A wrapper around broadcaster.Broadcast.

    Unlike the original implementation, this broadcaster will only connect
    when the first subscription is created and disconnect when the last
    subscription is removed.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._connect_task: asyncio.Task[Broadcast] | None = None
        self._broadcast: Broadcast | None = None
        self._subscriptions: list[Subscription] = []

    async def _connect_broadcast(self) -> Broadcast:
        """Connect a new broadcast, save it and return it."""
        broadcast = Broadcast(self.url)
        await broadcast.connect()
        self._broadcast = broadcast
        return broadcast

    async def _get_or_create_broadcaster(self) -> Broadcast:
        """Return the current broadcast or create a new one and connect it before returning it.

        I did not want to use a lock, because it's easy to forget to release it.
        So I use a task to connect the broadcast and save it in the attribute.
        Because asyncio is not parallel, but concurrent, there won't be two
        tasks trying to connect the broadcast at the same time.
        Either no connect task is running, then the task is created and awaited,
        or the task is already running, then the await will wait for the task to finish.
        """
        if broadcast := self._broadcast:
            return broadcast
        if connect_task := self._connect_task:
            if not connect_task.done():
                await asyncio.wait([connect_task])
            return connect_task.result()
        else:
            self._connect_task = asyncio.create_task(self._connect_broadcast())
            return await self._connect_task

    async def _remove_subscription(self, subscription: Subscription) -> None:
        try:
            self._subscriptions.remove(subscription)
        except ValueError:
            pass
        if not self._subscriptions and self._broadcast:
            broadcast = self._broadcast
            self._broadcast = None
            self._connect_task = None
            await broadcast.disconnect()

    async def _connect_subscription(self, subscription: Subscription) -> None:
        broadcaster = await self._get_or_create_broadcaster()
        provider = broadcaster.subscribe(subscription.channel)
        subscriber = await provider.__aenter__()
        subscription._context = provider
        subscription._subscriber = subscriber
        self._subscriptions.append(subscription)

    def subscribe(self, channel: str) -> Subscription:
        """Return a new subscription for the given channel.

        The subscription must be used as an asynchronous context manager
        """
        return Subscription(channel=channel, listener=self)
