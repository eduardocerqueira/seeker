#date: 2023-01-27T16:59:40Z
#url: https://api.github.com/gists/c54b267a2b39f5d466fa6f388ae06774
#owner: https://api.github.com/users/leemurus

import asyncio
import time


def sync_print_performance(func):
    def wrapper(*args, **kwargs):
        begin = time.monotonic()
        result = func(*args, **kwargs)
        duration = time.monotonic() - begin
        print(f'Time execution of {func.__name__}: {duration * 1000:.2f} ms')
        return result

    return wrapper


def async_print_performance(func):
    async def wrapper(*args, **kwargs):
        begin = time.monotonic()
        result = await func(*args, **kwargs)
        duration = time.monotonic() - begin
        print(f'Time execution of {func.__name__}: {duration * 1000:.2f} ms')
        return result

    return wrapper


def sync_mul(a: int, b: int):
    return a * b


async def async_mul(a: int, b: int):
    return a * b


@sync_print_performance
def execute_sync_funcs(exc_number: int):
    for i in range(exc_number):
        sync_mul(i, i * 2)


@async_print_performance
async def execute_async_coros(exc_number: int):
    for i in range(exc_number):
        await async_mul(i, i * 2)


@async_print_performance
async def execute_async_tasks(exc_number: int):
    for i in range(exc_number):
        await asyncio.create_task(async_mul(i, i * 2))


@async_print_performance
async def execute_async_all_coros(exc_number: int):
    tasks = []

    for i in range(exc_number):
        tasks.append(async_mul(i, i * 2))

    await asyncio.wait(tasks)


@async_print_performance
async def execute_async_all_tasks(exc_number: int):
    tasks = []

    for i in range(exc_number):
        tasks.append(asyncio.create_task(async_mul(i, i * 3)))

    await asyncio.wait(tasks)


async def main():
    exc_number = 10000

    execute_sync_funcs(exc_number)
    await execute_async_coros(exc_number)
    await execute_async_tasks(exc_number)
    await execute_async_all_coros(exc_number)
    await execute_async_all_tasks(exc_number)


if __name__ == '__main__':
    asyncio.run(main())
