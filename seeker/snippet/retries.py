#date: 2024-08-14T19:03:33Z
#url: https://api.github.com/gists/2cf4deba48c32c01fdf6c0ee5248a1d1
#owner: https://api.github.com/users/jmehnle

from contextlib import contextmanager

import asyncpg
import tenacity.asyncio
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_delay, wait_exponential


_sleep_enabled = True

# This indirection allows for global disabling of sleep, such as during unit tests.
async def _sleep(seconds: float) -> None:
    if _sleep_enabled:
        await tenacity.asyncio._portable_async_sleep(seconds)

@contextmanager
def no_retry_sleep():
    global _sleep_enabled
    _sleep_enabled = False
    try:
        yield
    finally:
        _sleep_enabled = True

def make_retry(*exception_types: type[Exception], **kwargs: Any) -> AsyncRetrying:
    kwargs = {
        'wait': wait_exponential(min=10, max=60),
        'stop': stop_after_delay(600),
        'sleep': _sleep,
        **kwargs,
    }
    if exception_types:
        kwargs['retry'] = retry_if_exception_type(exception_types)
    return AsyncRetrying(**kwargs)


pg_retry = make_retry(
    asyncpg.InterfaceError,
)

aws_retry = make_retry(
    ...
)

...
