#date: 2025-01-06T16:59:03Z
#url: https://api.github.com/gists/3b911dc84d9b6d7cdd6e69ac6bb1ba2c
#owner: https://api.github.com/users/XanderMoroz

import traceback
import sys
import inspect
import logging
import asyncio

# Настройка логирования
logging.basicConfig(level=logging.ERROR)

def custom_exception_handler(function):
    """
    Декоратор для обработки исключений в синхронных и асинхронных функциях.

    Этот декоратор перехватывает исключения, возникающие в декорируемой функции,
    и выводит информацию об ошибке, включая тип исключения, аргументы,
    трассировку и соответствующий код.

    Args::
        - function: Функция, которую нужно декорировать. Может быть синхронной или асинхронной.
    Returns:
        Обернутую функцию, которая обрабатывает исключения.
    """

    async def async_wrapper(*args, **kwargs):
        """Обертка для асинхронных функций"""
        try:
            return await function(*args, **kwargs)
        except Exception as e:
            handle_exception(e)

    def sync_wrapper(*args, **kwargs):
        """Обертка для синхронных функций"""
        try:
            return function(*args, **kwargs)
        except Exception as e:
            handle_exception(e)

    def handle_exception(e):
        """Обрабатывает исключение и выводит информацию о нем."""
        logging.error("Exception Type: %s", type(e).__name__)
        logging.error("Exception Args: %s", e.args)
        logging.error("Traceback:")
        traceback.print_tb(e.__traceback__)

        # Извлечение информации о последнем вызове
        stack = traceback.extract_tb(sys.exc_info()[2])
        last_call = stack[-1]
        line_number = last_call.lineno
        function_name = last_call.name
        source_code = inspect.getsourcelines(inspect.getmodule(inspect.currentframe()))[0]
        relevant_code = source_code[line_number-10: line_number]

        logging.error(f"Line {line_number}, in {function_name}: {''.join(relevant_code)}")

    # Проверяем, является ли функция асинхронной
    if inspect.iscoroutinefunction(function):
        return async_wrapper
    else:
        return sync_wrapper


import random
import asyncio

# Предполагается, что custom_exception_handler уже определен здесь

@custom_exception_handler
def sync_function():
    """Синхронная функция, которая вызывает исключение."""
    print("Выполнение синхронной функции...")
    raise ValueError("Ошибка в синхронной функции!")

@custom_exception_handler
async def async_function():
    """Асинхронная функция, которая вызывает исключение."""
    print("Выполнение асинхронной функции...")
    await asyncio.sleep(1)  # Имитация асинхронной работы
    raise TypeError("Ошибка в асинхронной функции!")

def main():
    # Проверка синхронной функции
    try:
        sync_function()
    except Exception:
        pass  # Исключение уже обработано декоратором

    # Проверка асинхронной функции
    try:
        asyncio.run(async_function())
    except Exception:
        pass  # Исключение уже обработано декоратором

if __name__ == "__main__":
    main()