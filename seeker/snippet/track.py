#date: 2022-06-27T16:57:31Z
#url: https://api.github.com/gists/58692cdd339766806c22469922b36c5f
#owner: https://api.github.com/users/k-nut

import urllib.request
import sqlite3
from time import sleep

URL = 'https://api.criticalmaps.net/postv2'
FILENAME = "criticaltracks.sqlite"


def get_data():
    with urllib.request.urlopen(URL) as url:
        return url.read().decode()


def add_row(cur, con, data: str):
    cur.execute('insert into tracks (data) values(?);', (data,))
    con.commit()


def main():
    con = sqlite3.connect(FILENAME)
    cur = con.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS tracks (timestamp datetime default current_timestamp, data text);')
    con.commit()

    while True:
        data = get_data()
        add_row(cur, con, data)
        sleep(5)
        print(".", end="")



if __name__ == "__main__":
    main()