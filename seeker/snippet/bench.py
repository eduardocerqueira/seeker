#date: 2024-08-30T16:36:13Z
#url: https://api.github.com/gists/6832dd94f8bb34a1a6a5a20de6af6132
#owner: https://api.github.com/users/samwho

import random
import string
import time
from typing import Tuple

import psycopg2
from psycopg2.extensions import cursor
from rich.console import Console
from rich.table import Table

LARGE_STRING = "a" * 64 * 1024


def random_string(length: int = 10) -> str:
    return "".join(random.choices(string.ascii_lowercase, k=length))


def create_tables(cur: cursor) -> None:
    cur.execute("""CREATE TABLE IF NOT EXISTS int_table
                   (id INTEGER PRIMARY KEY, value TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS string_table
                   (id TEXT PRIMARY KEY, value TEXT)""")


def truncate_table(cur: cursor, table_name: str) -> None:
    cur.execute(f"TRUNCATE TABLE {table_name}")
    cur.connection.commit()


def insert_data(
    cur: cursor, table_name: str, data: list[Tuple[int | str, str]]
) -> float:
    total = 0
    truncate_table(cur, table_name)
    for record in data:
        start_time = time.perf_counter()
        cur.execute(f"INSERT INTO {table_name} (id, value) VALUES (%s, %s)", record)
        cur.connection.commit()
        end_time = time.perf_counter()
        total += end_time - start_time
    return total


def read_data(cur: cursor, table_name: str, ids: list[int | str]) -> float:
    total = 0
    for id in ids:
        start_time = time.perf_counter()
        cur.execute(f"SELECT * FROM {table_name} WHERE id = %s", (id,))
        cur.fetchone()
        end_time = time.perf_counter()
        total += end_time - start_time
    return total


def benchmark(num_records: int = 10000, num_reads: int = 1000) -> dict[str, float]:
    # PostgreSQL connection parameters
    conn_params = {
        "dbname": "postgres",
        "user": "postgres",
        "password": "**********"
        "host": "localhost",
        "port": "5432",
    }

    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()

    # Ensure tables don't exist
    cur.execute("DROP TABLE IF EXISTS int_table")
    cur.execute("DROP TABLE IF EXISTS string_table")
    conn.commit()

    create_tables(cur)
    conn.commit()

    ints = list(range(num_records))
    random_ints = ints
    random.shuffle(random_ints)

    # Prepare data
    int_seq_data = [(i, LARGE_STRING) for i in ints]
    int_random_data = [(i, LARGE_STRING) for i in random_ints]
    str_seq_data = [(f"{i:010d}", LARGE_STRING) for i in ints]
    str_random_data = [(random_string(), LARGE_STRING) for i in ints]

    # Benchmark insertions
    int_seq_insert = insert_data(cur, "int_table", int_seq_data)
    int_random_insert = insert_data(cur, "int_table", int_random_data)
    str_seq_insert = insert_data(cur, "string_table", str_seq_data)
    str_random_insert = insert_data(cur, "string_table", str_random_data)

    # Prepare read data
    int_seq_ids = [i for i, _ in int_seq_data[:num_reads]]
    int_random_ids = [i for i, _ in int_random_data[:num_reads]]
    str_seq_ids = [i for i, _ in str_seq_data[:num_reads]]
    str_random_ids = [i for i, _ in str_random_data[:num_reads]]

    # Benchmark reads
    int_seq_read = read_data(cur, "int_table", int_seq_ids)
    int_random_read = read_data(cur, "int_table", int_random_ids)
    str_seq_read = read_data(cur, "string_table", str_seq_ids)
    str_random_read = read_data(cur, "string_table", str_random_ids)

    cur.close()
    conn.close()

    return {
        ("int", "sequential", "insert"): int_seq_insert,
        ("int", "random", "insert"): int_random_insert,
        ("str", "sequential", "insert"): str_seq_insert,
        ("str", "random", "insert"): str_random_insert,
        ("int", "sequential", "read"): int_seq_read,
        ("int", "random", "read"): int_random_read,
        ("str", "sequential", "read"): str_seq_read,
        ("str", "random", "read"): str_random_read,
    }


if __name__ == "__main__":
    n = 10000
    results = benchmark(num_records=n, num_reads=n)

    table = Table()

    table.add_column("Type", style="cyan", no_wrap=True)
    table.add_column("Mode", style="cyan", no_wrap=True)
    table.add_column("Operation", style="cyan", no_wrap=True)
    table.add_column("Time (seconds)", style="magenta")

    for (type, mode, op), time in results.items():
        table.add_row(type, mode, op, f"{time:.3f}")

    console = Console()
    console.print(table)
table)
