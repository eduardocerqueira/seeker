#date: 2024-06-26T16:33:43Z
#url: https://api.github.com/gists/43e10f4963089d5b7bb7373235672bb5
#owner: https://api.github.com/users/denisxab

import re
import traceback
from functools import wraps
from typing import Any

from django.db.backends.utils import CursorWrapper


class TraceSQL:
    """Класс для трассировки SQL запросов.
    Позволяет узнать в какой части кода выполнялся SQL запрос, для того чтобы его оптимизировать.

    Обычно вы можете узнать только какие SQL запросы выполнялись, но не знаете в какой части кода они выполнились,
    и тогда нужно тратить время и угадывать какая строка кода вызывает эти SQL запросы.

    Пример использования:

    class ИмяApiViewTests(APITestCase):
        def test_имя(self) -> None:
            # В path_trace_compile укажем регулярное выражение для фильтрации стека трассировки
            with TraceSQL(path_trace_compile="/code/api/(?!.*tests).*") as trace:
                response = self.client.put(
                    'URL',
                    data={
                        "имя": "имя",
                        "пол": "мужской",
                        "дата рождения": "01.01.2000",
                    },
                    format="json",
                )
            # Трассировка выполненных запросов
            print(trace.stack_sql)
    """

    def __init__(self, path_trace_compile: str):
        """Инициализация.

        Args:
            path_trace_compile: Регулярное выражение для фильтрации файлов, которые мы хотим трассировать.
            Например, укажите `/api/*`, чтобы игнорировать в трассировке файлы библиотек и тестов.
        """
        self.path_trace_compile = re.compile(path_trace_compile)
        # Список запросов
        self.stack_sql: list[dict[str, Any]] = []
        # Сохраним оригинальные функции
        self.original_execute = CursorWrapper.execute
        self.original_executemany = CursorWrapper.executemany

    def __enter__(self):
        def base_trace_stack(sql: str, params: list):
            stack: traceback.StackSummary = traceback.extract_stack()[:-1]
            filter_stack: list[traceback.FrameSummary] = [
                frame for frame in stack if self.path_trace_compile.search(frame.filename)
            ]
            self.stack_sql.append({"sql": sql, "stack": filter_stack, "params": params})

        @wraps(CursorWrapper.executemany)
        def execute_with_trace(cursor, sql, params=None):
            """Трассировка запросов для execute."""
            base_trace_stack(sql, params)
            return self.original_execute(cursor, sql, params)

        @wraps(CursorWrapper.executemany)
        def executemany_with_trace(cursor, sql, param_list):
            """Трассировка запросов для executemany."""
            base_trace_stack(sql, param_list)
            return self.original_executemany(cursor, sql, param_list)

        # Модифицируем функции для трассировки
        CursorWrapper.execute = execute_with_trace
        CursorWrapper.executemany = executemany_with_trace

        return self

    def __exit__(self, type, value, traceback):
        """Возвращает оригинальные функции."""
        CursorWrapper.execute = self.original_execute
        CursorWrapper.executemany = self.original_executemany
