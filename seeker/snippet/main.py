#date: 2024-01-24T17:05:30Z
#url: https://api.github.com/gists/034ab39e2a76b91ad0e07c08dc8b1a7e
#owner: https://api.github.com/users/goabonga

#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This script demonstrates the use of the aredis library for asynchronous Redis operations in Python.
It includes functions for subscribing to and publishing messages on a Redis channel.
The script performs the following actions:
- Subscribes to a Redis channel named 'foo'.
- Publishes two messages to the 'foo' channel.
- Waits for and prints any messages received on the 'foo' channel.
"""

import aredis
import asyncio
import concurrent.futures
import time
import logging


async def wait_for_message(pubsub, timeout=2, ignore_subscribe_messages=False):
    """
    Asynchronously waits for a message on the subscribed Redis channel.

    :param pubsub: Aredis PubSub object.
    :param timeout: Timeout in seconds for waiting for a message. Defaults to 2 seconds.
    :param ignore_subscribe_messages: If True, ignores subscription messages. Defaults to False.
    :return: None. Prints messages if received within the timeout period.
    """
    now = time.time()
    timeout = now + timeout
    while now < timeout:
        message = await pubsub.get_message(
            ignore_subscribe_messages=ignore_subscribe_messages,
            timeout=1
        )
        if message is not None:
            print(message)
        await asyncio.sleep(0.01)
        now = time.time()
    return None


async def subscribe(client):
    """
    Subscribes to the 'foo' channel and waits for messages.

    :param client: Aredis client instance.
    """
    await client.flushdb()
    pubsub = client.pubsub()
    assert pubsub.subscribed is False
    await pubsub.subscribe('foo')
    await wait_for_message(pubsub)


async def publish(client):
    """
    Publishes messages to the 'foo' channel.

    :param client: Aredis client instance.
    """
    # Sleep to wait for subscriber to listen
    await asyncio.sleep(1)
    await client.publish('foo', 'test message')
    await client.publish('foo', 'quit')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    client = aredis.StrictRedis()
    loop = asyncio.get_event_loop()

    loop.set_debug(enabled=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(asyncio.run_coroutine_threadsafe, publish(client), loop)

    loop.run_until_complete(subscribe(client))
