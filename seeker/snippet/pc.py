#date: 2021-10-01T16:59:35Z
#url: https://api.github.com/gists/7d756e4d8363a6bb9868effde1e14e3c
#owner: https://api.github.com/users/ca0abinary

# Requires python3.9 and
# pip3.9 install --user requests pyodbc threading

# Outputs results.csv

import requests
import pyodbc
import threading
import queue
from time import sleep

q = queue.Queue()
lock = threading.Lock()


def main() -> None:
    rows = [{'uri': 'google.com'},{'uri': 'yahoo.com'}]

    with open('results.csv', 'wt') as f:
        print(f'Got {len(rows)} records, analyzing...\n')

        f.write('status_code,id,uri,content,fix\n')

        rownum = 0
        for row in rows:
            uri = row.uri
            rownum += 1
            q.put(item={'id': rownum, 'uri': row.uri, 'file': f})

        for _ in range(64):
            threading.Thread(target=worker, daemon=True).start()

        while q.unfinished_tasks:
            print(f'{q.unfinished_tasks} records remaining')
            sleep(5)


def worker() -> None:
    while True:
        try:
            item = q.get(block=False)
        except queue.Empty:
            return

        fix = 're-publish'
        with requests.Session() as s:
            r = s.get(item['uri'])
            content = r.content[:4]

            if r.status_code == 200:
                if content == b'\x89PNG':
                    fix = ''
                elif content == b'\xFF\xD8\xFF\xE0':
                    fix = ''
                elif content == b'/var':
                    fix = 'rewrite'
                    content = r.content

            lock.acquire()
            item['file'].write(f'{r.status_code=},{item["id"]=},')
            item['file'].write(f'{item["uri"]=},{content=},{fix=}\n')
            lock.release()

        q.task_done()

if __name__ == "__main__":
    main()
