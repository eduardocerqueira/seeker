#date: 2021-10-01T16:50:26Z
#url: https://api.github.com/gists/7697ead59fc34a017043a1e13b9e6703
#owner: https://api.github.com/users/acu192

import sys
import time
import requests
from threading import Thread


URL = 'http://localhost:8003/api/temp/sleep'


def get():
    req = requests.get(URL)
    res = req.json()
    assert res['success']
    return res


def main(n):
    begin = time.time()
    threads = [Thread(target=get) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    end = time.time()
    print(f'Took: {end - begin:.2} seconds')


if __name__ == '__main__':
    main(int(sys.argv[1]))
