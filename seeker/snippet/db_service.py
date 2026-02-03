#date: 2026-02-03T17:46:07Z
#url: https://api.github.com/gists/5b07cd19f4d9a42c6f71d7c6f2b292cc
#owner: https://api.github.com/users/datavudeja

import os
import sqlite3
import unittest
from typing import Dict, List, Union, Any, Optional, Tuple
import threading
import queue
from concurrent.futures import Future
import asyncio


class ThreadSafeDBOperations:
    """
    A thread-safe wrapper for AdvancedDBOperations that processes
    database requests in a dedicated worker thread.
    Methods return a Future object. Call .result() on the future
    to get the actual return value.
    """

    def __init__(self, file_path: str):
        self.request_queue = queue.Queue()
        self.db_worker = None
        self.worker_thread = threading.Thread(target=self._worker_loop, args=(file_path,))
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def _worker_loop(self, file_path: str):
        """The main loop for the worker thread."""
        self.db_worker = AdvancedDBOperations(file_path)
        print(f'Worker thread started for DB: {file_path}')
        while True:
            request = self.request_queue.get()
            if request is None:
                break
            method_name, args, kwargs, future = request
            try:
                method = getattr(self.db_worker, method_name)
                result = method(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
        print('Worker thread stopped.')

    def __getattr__(self, name):
        """
        Перехватывает вызовы методов, упаковывает их в запрос
        и помещает в очередь. Это ядро всей магии.
        """

        def method(*args, **kwargs):
            future = Future()
            request = (name, args, kwargs, future)
            self.request_queue.put(request)
            return future

        return method

    def close(self):
        """Сигнализирует рабочему потоку о завершении работы."""
        if hasattr(self, '_closed') and self._closed:
            return
        if self.worker_thread.is_alive():
            self.request_queue.put(None)
            self.worker_thread.join()
        self._closed = True

    def __del__(self):
        self.close()


class AdvancedDBOperations:
    """
    A class for performing advanced database operations using SQLite.
    """

    def __init__(self, file_path: str):
        """
        Initializes the AdvancedDBOperations object.
        Args:
            file_path (str): The path to the SQLite database file.
        """
        self.file_path: str = file_path
        self.conn: sqlite3.Connection = sqlite3.connect(self.file_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor: sqlite3.Cursor = self.conn.cursor()

    def create_table(self, table_name: str, fields: Dict[str, str]) -> None:
        """
        Creates a table with the specified name and fields.
        Args:
            table_name (str): The name of the table to create.
            fields (Dict[str, str]): A dictionary where keys are field names and values are field data types.
        """
        fields_str: str = ', '.join([f'{field} {dtype}' for field, dtype in fields.items()])
        self.cursor.execute(f'CREATE TABLE IF NOT EXISTS {table_name} ({fields_str})')
        self.conn.commit()

    def create_index(self, table_name: str, column_name: str, unique: bool = False) -> None:
        """
        Creates an index for the specified column in the given table.
        Args:
            table_name (str): The name of the table.
            column_name (str): The name of the column to create an index for.
            unique (bool, optional): Whether to create a unique index. Defaults to False.
        """
        index_name: str = f'{table_name}_{column_name}_index'
        unique_str: str = 'UNIQUE' if unique else ''
        self.cursor.execute(f'CREATE {unique_str} INDEX IF NOT EXISTS {index_name} ON {table_name} ({column_name})')
        self.conn.commit()

    def drop_index(self, table_name: str, column_name: str) -> None:
        """
        Drops the index for the specified column in the given table.
        Args:
            table_name (str): The name of the table.
            column_name (str): The name of the column to drop the index for.
        """
        index_name: str = f'{table_name}_{column_name}_index'
        self.cursor.execute(f'DROP INDEX IF EXISTS {index_name}')
        self.conn.commit()

    def insert(self, table_name: str, data: Dict[str, Union[str, int, float, None]]) -> Optional[int]:
        """
        Inserts data into the specified table.
        Args:
            table_name (str): The name of the table.
            data (Dict[str, Union[str, int, float, None]]): A dictionary where keys are column names and values are the data to insert.
        Returns:
            Optional[int]: The row ID of the inserted row, or None if an error occurred.
        """
        try:
            placeholders: str = ', '.join(['?' for _ in data])
            columns: str = ', '.join(data.keys())
            values: Tuple[Any, ...] = tuple(data.values())
            self.cursor.execute(f'INSERT INTO {table_name} ({columns}) VALUES ({placeholders})', values)
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            print(f'Error inserting data into table {table_name}: {e}')
            return None
        except (TypeError, KeyError) as e:
            print(f'Error inserting data: Invalid data format: {e}')
            return None

    def edit(self, table_name: str, id: int, data: Dict[str, Union[str, int, float, None]]) -> int:
        """
        Edits a record in the specified table by ID.
        Args:
            table_name (str): The name of the table.
            id (int): The ID of the record to edit.
            data (Dict[str, Union[str, int, float, None]]): A dictionary where keys are column names and values are the new data.
        Returns:
            int: The number of rows affected.
        """
        try:
            set_clause: str = ', '.join([f'{key} = ?' for key in data.keys()])
            values: Tuple[Any, ...] = tuple(data.values()) + (id,)
            self.cursor.execute(f'UPDATE {table_name} SET {set_clause} WHERE id = ?', values)
            self.conn.commit()
            if self.cursor.rowcount == 0:
                print(f'Warning: No record found with ID {id} in table {table_name}.')
            return self.cursor.rowcount
        except sqlite3.Error as e:
            print(f'Error editing data in table {table_name}: {e}')
            return 0
        except (TypeError, KeyError) as e:
            print(f'Error editing data: Invalid data format: {e}')
            return 0

    def delete(self, table_name: str, id: int) -> int:
        """
        Deletes a record from the specified table by ID.
        Args:
            table_name (str): The name of the table.
            id (int): The ID of the record to delete.
        Returns:
            int: The number of rows affected.
        """
        try:
            self.cursor.execute(f'DELETE FROM {table_name} WHERE id = ?', (id,))
            self.conn.commit()
            if self.cursor.rowcount == 0:
                print(f'Warning: No record found with ID {id} in table {table_name}.')
            return self.cursor.rowcount
        except sqlite3.Error as e:
            print(f'Error deleting data from table {table_name}: {e}')
            return 0

    def select(
        self,
        table_name: str,
        params: Optional[Dict[str, Any]] = None,
        joins: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs a SELECT query on the specified table with optional filtering and joins.
        Args:
            table_name (str): The name of the table.
            params (Optional[Dict[str, Any]], optional): A dictionary of parameters for filtering the data.
                Supported parameters:
                - date_range (Tuple[str, str]): A tuple containing the start and end dates for filtering by date.
                - search (str): A string to search for in text and varchar fields.
                - ids (List[int]): A list of IDs to select.
                - exact_match (Dict[str, Dict[str, Any]]): A dictionary for exact matching of fields.
                - sort (Dict[str, str]): A dictionary specifying the field and order for sorting.
                - limit (int): The maximum number of rows to return.
                - offset (int): The number of rows to skip.
            joins (Optional[List[Dict[str, str]]], optional): A list of dictionaries specifying joins with other tables.
                Each dictionary should have the following keys:
                - type (str): The type of join (e.g., "INNER", "LEFT").
                - table (str): The name of the table to join with.
                - on (str): The join condition.
        Returns:
            List[Dict[str, Any]]: A list of rows matching the query.
        """
        try:
            query: str = f'SELECT * FROM {table_name}'
            conditions: List[str] = []
            values: List[Any] = []
            if joins:
                for join in joins:
                    query += f" {join['type']} JOIN {join['table']} ON {join['on']}"
            if params:
                if 'date_range' in params:
                    start, end = params['date_range']
                    conditions.append(f'{table_name}.date BETWEEN ? AND ?')
                    values.extend([start, end])
                if 'search' in params:
                    search_conditions: List[str] = []
                    for field in self.get_table_fields(table_name):
                        if field.lower() == 'name':
                            search_conditions.append(f'{table_name}.{field} LIKE ?')
                            values.append(f"%{params['search']}%")
                    if search_conditions:
                        conditions.append(f"({' OR '.join(search_conditions)})")
                if 'ids' in params:
                    placeholders: str = ', '.join(['?' for _ in params['ids']])
                    conditions.append(f'{table_name}.id IN ({placeholders})')
                    values.extend(params['ids'])
                if 'exact_match' in params:
                    for table, match_data in params['exact_match'].items():
                        for field, value in match_data.items():
                            conditions.append(f'{table}.{field} = ?')
                            values.append(value)
                if conditions:
                    query += ' WHERE ' + ' AND '.join(conditions)
                if 'sort' in params:
                    query += f" ORDER BY {params['sort']['field']} {params['sort']['order']}"
                if 'limit' in params:
                    query += f" LIMIT {params['limit']}"
                if 'offset' in params:
                    query += f" OFFSET {params['offset']}"
            self.cursor.execute(query, tuple(values))
            rows = self.cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f'Error selecting data from table {table_name}: {e}')
            return []
        except (TypeError, KeyError) as e:
            print(f'Error selecting data: Invalid data format: {e}')
            return []

    def aggregate(
        self, table_name: str, func: str, field: str, params: Optional[Dict[str, Any]] = None
    ) -> Union[float, int]:
        """
        Performs aggregate functions (SUM, AVG, COUNT, MIN, MAX) on data in the table.
        Args:
            table_name (str): The name of the table.
            func (str): The name of the aggregate function (SUM, AVG, COUNT, MIN, MAX).
            field (str): The field on which to perform the aggregate function.
            params (Optional[Dict[str, Any]], optional): Additional parameters for filtering data.
                Supported parameters:
                - date_range (Tuple[str, str]): A tuple containing the start and end dates for filtering by date.
                - search (str): A string to search for in text and varchar fields.
                - ids (List[int]): A list of IDs to select.
                - exact_match (Dict[str, Any]): A dictionary for exact matching of fields.
        Returns:
            Union[float, int]: The result of the aggregate function.
        """
        try:
            func = func.upper()
            if func not in ('SUM', 'AVG', 'COUNT', 'MIN', 'MAX'):
                raise ValueError('Invalid aggregate function. Choose from SUM, AVG, COUNT, MIN, MAX.')
            query: str = f'SELECT {func}({field}) FROM {table_name}'
            conditions: List[str] = []
            values: List[Any] = []
            if params:
                if 'date_range' in params:
                    start, end = params['date_range']
                    conditions.append('date BETWEEN ? AND ?')
                    values.extend([start, end])
                if 'search' in params:
                    search_conditions: List[str] = []
                    for f in self.get_table_fields(table_name):
                        if f.lower() in ('text', 'varchar'):
                            search_conditions.append(f'{f} LIKE ?')
                            values.append(f"%{params['search']}%")
                    if search_conditions:
                        conditions.append(f"({' OR '.join(search_conditions)})")
                if 'ids' in params:
                    placeholders: str = ', '.join(['?' for _ in params['ids']])
                    conditions.append(f'id IN ({placeholders})')
                    values.extend(params['ids'])
                if 'exact_match' in params:
                    for f, value in params['exact_match'].items():
                        conditions.append(f'{f} = ?')
                        values.append(value)
                if conditions:
                    query += ' WHERE ' + ' AND '.join(conditions)
            self.cursor.execute(query, tuple(values))
            result: Optional[Union[float, int]] = self.cursor.fetchone()[0]
            return result if result is not None else 0
        except sqlite3.Error as e:
            print(f'Error performing aggregate function: {e}')
            return 0
        except (TypeError, KeyError, ValueError) as e:
            print(f'Error performing aggregate function: {e}')
            return 0

    def group_by(
        self, table_name: str, group_by_fields: List[str], params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Union[str, int]]]:
        """
        Performs a GROUP BY query on the specified table.
        Args:
            table_name (str): The name of the table.
            group_by_fields (List[str]): A list of fields to group by.
            params (Optional[Dict[str, Any]], optional): Additional parameters for filtering and sorting data.
                Supported parameters:
                - date_range (Tuple[str, str]): A tuple containing the start and end dates for filtering by date.
                - search (str): A string to search for in text and varchar fields.
                - ids (List[int]): A list of IDs to select.
                - exact_match (Dict[str, Any]): A dictionary for exact matching of fields.
                - sort (Dict[str, str]): A dictionary specifying the field and order for sorting.
        Returns:
            List[Dict[str, Union[str, int]]]: A list of rows resulting from the GROUP BY query.
        """
        try:
            query: str = f"SELECT {', '.join(group_by_fields)}, COUNT(*) as count FROM {table_name}"
            conditions: List[str] = []
            values: List[Any] = []
            if params:
                if 'date_range' in params:
                    start, end = params['date_range']
                    conditions.append('date BETWEEN ? AND ?')
                    values.extend([start, end])
                if 'search' in params:
                    search_conditions: List[str] = []
                    for field in self.get_table_fields(table_name):
                        if field.lower() in ('text', 'varchar'):
                            search_conditions.append(f'{field} LIKE ?')
                            values.append(f"%{params['search']}%")
                    if search_conditions:
                        conditions.append(f"({' OR '.join(search_conditions)})")
                if 'ids' in params:
                    placeholders: str = ', '.join(['?' for _ in params['ids']])
                    conditions.append(f'id IN ({placeholders})')
                    values.extend(params['ids'])
                if 'exact_match' in params:
                    for field, value in params['exact_match'].items():
                        conditions.append(f'{field} = ?')
                        values.append(value)
                if conditions:
                    query += ' WHERE ' + ' AND '.join(conditions)
            query += f" GROUP BY {', '.join(group_by_fields)}"
            if params and 'sort' in params and params['sort']['field'] == 'count':
                query += f" ORDER BY count {params['sort']['order']}"
            elif params and 'sort' in params:
                query += f" ORDER BY {params['sort']['field']} {params['sort']['order']}"
            self.cursor.execute(query, tuple(values))
            rows = self.cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f'Error performing GROUP BY query on table {table_name}: {e}')
            return []
        except (TypeError, KeyError) as e:
            print(f'Error performing GROUP BY query: Invalid data format: {e}')
            return []

    def get_table_fields(self, table_name: str) -> List[str]:
        """
        Returns a list of fields for the specified table.
        Args:
            table_name (str): The name of the table.
        Returns:
            List[str]: A list of field names.
        """
        self.cursor.execute(f'PRAGMA table_info({table_name})')
        return [row[1] for row in self.cursor.fetchall()]

    def begin_transaction(self) -> None:
        """Begins a transaction."""
        self.conn.execute('BEGIN')

    def commit_transaction(self) -> None:
        """Commits the current transaction."""
        self.conn.commit()

    def rollback_transaction(self) -> None:
        """Rolls back the current transaction."""
        self.conn.rollback()

    def __del__(self):
        """Closes the database connection when the object is deleted."""
        self.conn.close()


class TestAdvancedDBOperations(unittest.TestCase):
    TEST_DB_PATH = ':memory:'

    def setUp(self):
        self.db = AdvancedDBOperations(self.TEST_DB_PATH)
        self.db.create_table(
            'users',
            {'id': 'INTEGER PRIMARY KEY', 'name': 'TEXT', 'age': 'INTEGER', 'city_id': 'INTEGER', 'date': 'TEXT'},
        )
        self.db.create_table('cities', {'id': 'INTEGER PRIMARY KEY', 'name': 'TEXT'})
        self.db.create_table(
            'orders',
            {
                'id': 'INTEGER PRIMARY KEY',
                'user_id': 'INTEGER',
                'product': 'TEXT',
                'price': 'REAL',
                'order_date': 'TEXT',
            },
        )
        users_data = [
            {'name': 'John Doe', 'age': 30, 'city_id': 1, 'date': '2023-05-15'},
            {'name': 'Jane Smith', 'age': 25, 'city_id': 2, 'date': '2023-06-20'},
            {'name': 'Edvard', 'age': 31, 'city_id': 2, 'date': '2023-09-21'},
            {'name': 'Alice Johnson', 'age': 35, 'city_id': 1, 'date': '2023-07-10'},
            {'name': 'Bob Williams', 'age': 40, 'city_id': 3, 'date': '2023-08-05'},
        ]
        for user in users_data:
            self.db.insert('users', user)
        cities_data = [
            {'name': 'New York'},
            {'name': 'Los Angeles'},
            {'name': 'Chicago'},
        ]
        for city in cities_data:
            self.db.insert('cities', city)
        orders_data = [
            {'user_id': 1, 'product': 'Laptop', 'price': 1200.00, 'order_date': '2023-05-20'},
            {'user_id': 2, 'product': 'Mouse', 'price': 25.00, 'order_date': '2023-06-25'},
            {'user_id': 1, 'product': 'Keyboard', 'price': 75.00, 'order_date': '2023-07-15'},
            {'user_id': 4, 'product': 'Monitor', 'price': 300.00, 'order_date': '2023-07-18'},
            {'user_id': 5, 'product': 'USB Drive', 'price': 15.00, 'order_date': '2023-08-10'},
            {'user_id': 3, 'product': 'Laptop', 'price': 1500.00, 'order_date': '2023-10-10'},
        ]
        for order in orders_data:
            self.db.insert('orders', order)

    def tearDown(self):
        self.db.conn.close()
        if os.path.exists(self.TEST_DB_PATH):
            os.remove(self.TEST_DB_PATH)

    def test_create_table(self):
        self.db.create_table('test_table', {'id': 'INTEGER PRIMARY KEY', 'value': 'TEXT'})
        self.assertTrue(
            'test_table'
            in [
                row[0] for row in self.db.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            ]
        )

    def test_create_index(self):
        self.db.create_index('users', 'name')
        indexes = self.db.cursor.execute(f"PRAGMA index_list('users')").fetchall()
        self.assertTrue(any('users_name_index' in index[1] for index in indexes))

    def test_create_unique_index(self):
        self.db.create_index('cities', 'name', unique=True)
        indexes = self.db.cursor.execute(f"PRAGMA index_list('cities')").fetchall()
        self.assertTrue(any('cities_name_index' in index[1] for index in indexes))
        self.assertTrue(any(index[2] == 1 for index in indexes if index[1] == 'cities_name_index'))

    def test_drop_index(self):
        self.db.create_index('users', 'name')
        self.db.drop_index('users', 'name')
        indexes = self.db.cursor.execute(f"PRAGMA index_list('users')").fetchall()
        self.assertFalse(any('users_name_index' in index[1] for index in indexes))

    def test_insert(self):
        id = self.db.insert('users', {'name': 'Test User', 'age': 25, 'city_id': 1, 'date': '2024-01-01'})
        self.assertIsNotNone(id)
        result = self.db.select('users', params={'ids': [id]})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['name'], 'Test User')

    def test_edit(self):
        self.db.edit('users', 1, {'age': 31})
        result = self.db.select('users', params={'ids': [1]})
        self.assertEqual(result[0]['age'], 31)

    def test_edit_nonexistent(self):
        rows_affected = self.db.edit('users', 100, {'age': 31})
        self.assertEqual(rows_affected, 0)

    def test_delete(self):
        self.db.delete('users', 1)
        result = self.db.select('users', params={'ids': [1]})
        self.assertEqual(len(result), 0)

    def test_delete_nonexistent(self):
        rows_affected = self.db.delete('users', 100)
        self.assertEqual(rows_affected, 0)

    def test_select_date_range(self):
        result = self.db.select('users', params={'date_range': ('2023-07-01', '2023-12-31')})
        self.assertEqual(len(result), 3)

    def test_select_search(self):
        result = self.db.select('users', params={'search': 'John'})
        self.assertEqual(len(result), 2)

    def test_select_ids(self):
        result = self.db.select('users', params={'ids': [1, 3, 5]})
        self.assertEqual(len(result), 3)

    def test_select_exact_match(self):
        result = self.db.select('users', params={'exact_match': {'users': {'age': 35}}})
        self.assertEqual(len(result), 1)

    def test_select_sort(self):
        result = self.db.select('users', params={'sort': {'field': 'date', 'order': 'DESC'}})
        self.assertEqual(result[0]['name'], 'Edvard')

    def test_select_limit(self):
        result = self.db.select('users', params={'limit': 3})
        self.assertEqual(len(result), 3)

    def test_select_offset(self):
        result = self.db.select('users', params={'offset': 2, 'limit': 2})
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['name'], 'Edvard')

    def test_select_join(self):
        result = self.db.select(
            'users',
            joins=[{'type': 'INNER', 'table': 'orders', 'on': 'users.id = orders.user_id'}],
            params={
                'sort': {'field': 'users.id', 'order': 'ASC'},
                'limit': 2,
            },
        )
        self.assertEqual(len(result), 2)

    def test_aggregate_sum(self):
        result = self.db.aggregate('orders', 'SUM', 'price')
        self.assertEqual(result, 3115.0)

    def test_aggregate_avg(self):
        result = self.db.aggregate('orders', 'AVG', 'price')
        self.assertEqual(result, 519.1666666666666)

    def test_aggregate_count(self):
        result = self.db.aggregate('orders', 'COUNT', 'id')
        self.assertEqual(result, 6)

    def test_aggregate_min(self):
        result = self.db.aggregate('orders', 'MIN', 'price')
        self.assertEqual(result, 15.0)

    def test_aggregate_max(self):
        result = self.db.aggregate('orders', 'MAX', 'price')
        self.assertEqual(result, 1500.0)

    def test_aggregate_invalid_function(self):
        result = self.db.aggregate('orders', 'INVALID', 'price')
        self.assertEqual(result, 0)

    def test_group_by(self):
        result = self.db.group_by('users', ['city_id'], params={'sort': {'field': 'count', 'order': 'DESC'}})
        self.assertEqual(len(result), 3)
        self.assertEqual(result[1]['city_id'], 1)
        self.assertEqual(result[1]['count'], 2)

    def test_commit_transaction(self):
        self.db.begin_transaction()
        self.db.insert('users', {'name': 'Test User', 'age': 25, 'city_id': 1, 'date': '2024-01-01'})
        self.db.commit_transaction()
        result = self.db.select('users', params={'search': 'Test User'})
        self.assertEqual(len(result), 1)


class TestThreadSafeDBOperations(unittest.TestCase):
    TEST_DB_PATH = ':memory:'

    def setUp(self):
        """
        Инициализируем потокобезопасную обертку и наполняем БД.
        Поскольку все операции асинхронные, мы должны дождаться их завершения.
        """
        self.db = ThreadSafeDBOperations(self.TEST_DB_PATH)
        futures = [
            self.db.create_table('users', {'id': 'INTEGER PRIMARY KEY', 'name': 'TEXT', 'age': 'INTEGER'}),
            self.db.create_table('cities', {'id': 'INTEGER PRIMARY KEY', 'name': 'TEXT'}),
        ]
        for f in futures:
            f.result()
        users_data = [
            {'name': 'John Doe', 'age': 30},
            {'name': 'Jane Smith', 'age': 25},
        ]
        insert_futures = [self.db.insert('users', user) for user in users_data]
        for f in insert_futures:
            f.result()

    def tearDown(self):
        """Корректно останавливаем рабочий поток."""
        self.db.close()

    def test_insert_and_get_result(self):
        """Тестируем вставку и получение результата через future."""
        future = self.db.insert('users', {'name': 'Test User', 'age': 99})
        new_id = future.result()
        self.assertIsNotNone(new_id)
        select_future = self.db.select('users', params={'ids': [new_id]})
        result = select_future.result()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['name'], 'Test User')

    def test_select_search(self):
        """Тестируем поиск."""
        future = self.db.select('users', params={'search': 'John'})
        result = future.result()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['name'], 'John Doe')

    def test_edit(self):
        """Тестируем редактирование записи."""
        user_future = self.db.select('users', params={'exact_match': {'users': {'name': 'John Doe'}}})
        user_id = user_future.result()[0]['id']
        edit_future = self.db.edit('users', user_id, {'age': 31})
        rows_affected = edit_future.result()
        self.assertEqual(rows_affected, 1)
        check_future = self.db.select('users', params={'ids': [user_id]})
        result = check_future.result()
        self.assertEqual(result[0]['age'], 31)


class TestForumAPI(unittest.TestCase):
    TEST_DB_PATH = ':memory:'

    def setUp(self):
        """Создаем таблицы и наполняем их данными для тестов форума."""
        self.db = AdvancedDBOperations(self.TEST_DB_PATH)
        self.db.create_table('users', {'id': 'INTEGER PRIMARY KEY', 'name': 'TEXT', 'age': 'INTEGER'})
        users_data = [
            {'name': 'John Doe', 'age': 30},
            {'name': 'Jane Smith', 'age': 25},
        ]
        for user in users_data:
            self.db.insert('users', user)
        self.db.create_table('forums', {'id': 'INTEGER PRIMARY KEY', 'name': 'TEXT UNIQUE'})
        self.db.create_table(
            'posts',
            {
                'id': 'INTEGER PRIMARY KEY',
                'title': 'TEXT',
                'content': 'TEXT',
                'user_id': 'INTEGER',
                'forum_id': 'INTEGER',
                'post_date': 'TEXT',
            },
        )
        forums_data = [{'name': 'General Discussion'}, {'name': 'Tech Talk'}]
        for forum in forums_data:
            self.db.insert('forums', forum)
        posts_data = [
            {
                'title': 'Hello World',
                'content': 'My first post!',
                'user_id': 1,
                'forum_id': 1,
                'post_date': '2024-01-01',
            },
            {
                'title': 'About Python',
                'content': 'Python is great.',
                'user_id': 1,
                'forum_id': 2,
                'post_date': '2024-01-02',
            },
            {
                'title': 'About SQL',
                'content': 'SQLite is cool.',
                'user_id': 2,
                'forum_id': 2,
                'post_date': '2024-01-03',
            },
        ]
        for post in posts_data:
            self.db.insert('posts', post)

    def test_create_new_forum(self):
        """Тест: Создание нового форума."""
        forum_id = self.db.insert('forums', {'name': 'Python Fans'})
        self.assertIsNotNone(forum_id)
        result = self.db.select('forums', params={'exact_match': {'forums': {'name': 'Python Fans'}}})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['name'], 'Python Fans')

    def test_edit_forum_name(self):
        """Тест: Изменение названия форума."""
        rows_affected = self.db.edit('forums', 1, {'name': 'General Chat'})
        self.assertEqual(rows_affected, 1)
        result = self.db.select('forums', params={'ids': [1]})
        self.assertEqual(result[0]['name'], 'General Chat')

    def test_delete_forum(self):
        """Тест: Удаление форума."""
        self.db.cursor.execute('DELETE FROM posts WHERE forum_id = ?', (1,))
        self.db.conn.commit()
        rows_affected = self.db.delete('forums', 1)
        self.assertEqual(rows_affected, 1)
        result = self.db.select('forums', params={'ids': [1]})
        self.assertEqual(len(result), 0)

    def test_create_new_post(self):
        """Тест: Создание нового поста."""
        post_id = self.db.insert(
            'posts',
            {'title': 'New Topic', 'content': 'Content here.', 'user_id': 2, 'forum_id': 1, 'post_date': '2024-02-01'},
        )
        self.assertIsNotNone(post_id)
        result = self.db.select('posts', params={'ids': [post_id]})
        self.assertEqual(result[0]['title'], 'New Topic')

    def test_edit_post_content(self):
        """Тест: Редактирование содержимого поста."""
        rows_affected = self.db.edit('posts', 1, {'content': 'This is my very first post!'})
        self.assertEqual(rows_affected, 1)
        result = self.db.select('posts', params={'ids': [1]})
        self.assertEqual(result[0]['content'], 'This is my very first post!')

    def test_delete_post(self):
        """Тест: Удаление поста."""
        rows_affected = self.db.delete('posts', 1)
        self.assertEqual(rows_affected, 1)
        result = self.db.select('posts', params={'ids': [1]})
        self.assertEqual(len(result), 0)

    def test_find_posts_by_user(self):
        """Тест: Поиск всех постов конкретного пользователя."""
        result = self.db.select('posts', params={'exact_match': {'posts': {'user_id': 1}}})
        self.assertEqual(len(result), 2)
        self.assertTrue(all(post['user_id'] == 1 for post in result))

    def test_find_posts_with_joins(self):
        """Тест: Получение постов с информацией о пользователе и форуме."""
        joins = [
            {'type': 'LEFT', 'table': 'users', 'on': 'posts.user_id = users.id'},
            {'type': 'LEFT', 'table': 'forums', 'on': 'posts.forum_id = forums.id'},
        ]
        params = {'exact_match': {'posts': {'id': 2}}}
        self.db.cursor.execute(
            """
            SELECT posts.title, users.name as author_name, forums.name as forum_name
            FROM posts
            LEFT JOIN users ON posts.user_id = users.id
            LEFT JOIN forums ON posts.forum_id = forums.id
            WHERE posts.id = ?
        """,
            (2,),
        )
        result = dict(self.db.cursor.fetchone())
        self.assertEqual(result['title'], 'About Python')
        self.assertEqual(result['author_name'], 'John Doe')
        self.assertEqual(result['forum_name'], 'Tech Talk')

    def test_create_new_user(self):
        """Тест: Создание нового пользователя."""
        user_id = self.db.insert('users', {'name': 'Peter Jones', 'age': 45})
        self.assertIsNotNone(user_id)
        result = self.db.select('users', params={'ids': [user_id]})
        self.assertEqual(result[0]['name'], 'Peter Jones')
        self.assertEqual(result[0]['age'], 45)

    def tearDown(self):
        self.db.conn.close()


class TestAsyncForumAPI(unittest.TestCase):
    TEST_DB_PATH = ':memory:'

    def setUp(self):
        self.db = ThreadSafeDBOperations(self.TEST_DB_PATH)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        create_users = self.db.create_table('users', {'id': 'INTEGER PRIMARY KEY', 'name': 'TEXT'})
        create_posts = self.db.create_table(
            'posts', {'id': 'INTEGER PRIMARY KEY', 'title': 'TEXT', 'user_id': 'INTEGER'}
        )
        create_users.result()
        create_posts.result()
        insert_user = self.db.insert('users', {'name': 'AsyncUser'})
        insert_post1 = self.db.insert('posts', {'title': 'First Post', 'user_id': 1})
        insert_post2 = self.db.insert('posts', {'title': 'Second Post', 'user_id': 1})
        insert_user.result()
        insert_post1.result()
        insert_post2.result()

    def tearDown(self):
        self.db.close()
        self.loop.close()

    async def _await_future(self, future: Future):
        """
        Асинхронный хелпер для ожидания результата из concurrent.futures.Future.
        Он запускает блокирующий метод .result() в отдельном потоке.
        """
        return await self.loop.run_in_executor(None, future.result)

    def test_async_create_and_select_post(self):
        """Тест: Асинхронное создание и последующее чтение поста."""

        async def run_test():
            insert_future = self.db.insert('posts', {'title': 'Async Post', 'user_id': 1})
            new_id = await self._await_future(insert_future)
            self.assertIsNotNone(new_id)
            select_future = self.db.select('posts', params={'ids': [new_id]})
            result = await self._await_future(select_future)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['title'], 'Async Post')

        self.loop.run_until_complete(run_test())

    def test_async_concurrent_reads(self):
        """Тест: Запуск нескольких запросов на чтение одновременно."""

        async def run_test():
            future1 = self.db.select('posts', params={'ids': [1]})
            future2 = self.db.select('posts', params={'ids': [2]})
            results = await asyncio.gather(self._await_future(future1), self._await_future(future2))
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0][0]['title'], 'First Post')
            self.assertEqual(results[1][0]['title'], 'Second Post')

        self.loop.run_until_complete(run_test())

    def test_async_edit_and_verify(self):
        """Тест: Асинхронное редактирование и проверка."""

        async def run_test():
            edit_future = self.db.edit('users', 1, {'name': 'Updated AsyncUser'})
            rows_affected = await self._await_future(edit_future)
            self.assertEqual(rows_affected, 1)
            check_future = self.db.select('users', params={'ids': [1]})
            result = await self._await_future(check_future)
            self.assertEqual(result[0]['name'], 'Updated AsyncUser')

        self.loop.run_until_complete(run_test())


if __name__ == '__main__':
    unittest.main()
