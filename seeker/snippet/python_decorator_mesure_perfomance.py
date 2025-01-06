#date: 2025-01-06T16:42:58Z
#url: https://api.github.com/gists/77a3f5808392b7199a8039b4706cc4c3
#owner: https://api.github.com/users/XanderMoroz

import tracemalloc
from time import perf_counter
from typing import Callable, TypeVar, Awaitable
from functools import wraps

F = TypeVar('F', bound=Callable[..., Awaitable[None]])


def measure_performance(func: F) -> F:
	"""
    Асинхронный декоратор для измерения производительности функции.
	Attrs:
    	- func: Функция, производительность которой будет измеряться.
    Returns:
    	Обернутая функция с измерением времени и использования памяти.
    """
	@wraps(func)
	async def wrapper(*args, **kwargs) -> None:
		tracemalloc.start()
		start_time = perf_counter()

		# Вызов оригинальной функции
		await func(*args, **kwargs)

		current, peak = tracemalloc.get_traced_memory()
		finish_time = perf_counter()

		print(f'Function: {func.__name__}')
		print(f'Documentation: {func.__doc__}')
		print(f'Memory usage:\t\t {current / 10 ** 6:.6f} MB \n'
			  f'Peak memory usage:\t {peak / 10 ** 6:.6f} MB ')
		print(f'Time elapsed in seconds: {finish_time - start_time:.6f}')
		print(f'{"-" * 40}')

		tracemalloc.stop()

	return wrapper  # type: ignore


# Пример асинхронной функции с декоратором
@measure_performance
async def example_async_function() -> None:
	"""Пример асинхронной функции для тестирования производительности."""
	await asyncio.sleep(2)  # Имитация асинхронной работы

@measure_performance
async def example_async_function1():
	"""Range"""
	my_list = list(range(100000))


@measure_performance
async def example_async_function2():
	"""List comprehension"""
	my_list = [l for l in range(100000)]


# Пример вызова
async def main():
	await example_async_function()
	await example_async_function1()
	await example_async_function2()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())