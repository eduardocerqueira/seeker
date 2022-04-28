#date: 2022-04-28T17:09:59Z
#url: https://api.github.com/gists/661522923300a1fc657ac7480530faa7
#owner: https://api.github.com/users/edvardm

import math
import multiprocessing as mp

MAX_NUM_BATCHES = 8
BATCH_SIZE = 100_000


def consume(q):
    while not break_condition():
        if q.empty():
            nap("Sleeping for a while", seconds=math.pi)
        else:
            process_batch(q.get())


def break_condition():
    True  # replace with something more dynamic, by trapping signals etc


def run():
    q = mp.Queue(maxsize=MAX_NUM_BATCHES)
    db_conn = psycopg2.connect("postgresql://postgrest:sekrit@localhost/test")

    # create single producer, but would be simple to scale to multiple given reasonable way
    # to partition entries
    mp.Process(target=fetch_from_db, args=(db_conn, q)).start()

    # create single consumer, but could be simple to scale given this and that
    mp.Process(target=consume, args=(db_conn, q)).start()


def fetch_from_db(conn, q):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM muh_table")

    for batch in cursor.fetchmany(BATCH_SIZE):
        # blocks when queue full
        q.put(batch)


def process_batch(chunk):
    for row in chunk:
        print(f"processing row {row}")
