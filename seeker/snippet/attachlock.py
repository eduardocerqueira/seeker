#date: 2024-02-22T16:49:03Z
#url: https://api.github.com/gists/c57b3772ed5a36aabfe723df9820d6bc
#owner: https://api.github.com/users/lemon24

"""
Say you have a SQLite connection to database `main`
with attached database `attached`.

Does a long insert on the `attached` database in a similar connection
lock the `main` database in this connection?

No, it only locks `attached`.

Output:

one connects to 'main' and attaches 'attached'
two connects to 'attached' directly

two:
  insert into attached_table values (sleep(1))
one:
  insert into main_table values (1)

two:
  insert into attached_table values (sleep(1))
one:
  insert into attached_table values (1)
    error: database is locked

one connects to 'main' and attaches 'attached'
two connects to 'main' and attaches 'attached'

two:
  insert into attached_table values (sleep(1))
one:
  insert into main_table values (1)

two:
  insert into attached_table values (sleep(1))
one:
  insert into attached_table values (1)
    error: database is locked

"""
import os
import sqlite3
import tempfile
import time
import threading

mode = 'wal'

def do_stuff(attach):
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)

        one = sqlite3.connect('main', timeout=.1)
        one.execute(f"pragma journal_mode = {mode}")
        one.execute("attach 'attached' as attached")
        with one:
            one.execute("create table main_table(a)")
            one.execute("create table attached.attached_table(a)")

        print("one connects to 'main' and attaches 'attached'")
        if attach:
            print("two connects to 'main' and attaches 'attached'")
            two = sqlite3.connect('main', timeout=.1, check_same_thread=False)
            two.execute("attach 'attached' as attached")
        else:
            print("two connects to 'attached' directly")
            two = sqlite3.connect('attached', timeout=.1, check_same_thread=False)
        two.create_function('sleep', 1, time.sleep)

        print()

        names = {v: k for k, v in locals().items()}

        def execute(db, sql):
            print(f"{names[db]}:")
            print(f"  {sql}")
            try:
                db.execute(sql)
            except sqlite3.Error as e:
                print(f"    error: {e}")

        def execute_in_thread(db, sql):
            threading.Thread(target=execute, args=(db, sql)).start()

        with one, two:
            execute_in_thread(two, "insert into attached_table values (sleep(1))")
            time.sleep(.2)
            execute(one, "insert into main_table values (1)")
        print()
    
        with one, two:
            execute_in_thread(two, "insert into attached_table values (sleep(1))")
            time.sleep(.2)
            execute(one, "insert into attached_table values (1)")
        print()


do_stuff(False)    
do_stuff(True)    

