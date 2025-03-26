#date: 2025-03-26T17:11:01Z
#url: https://api.github.com/gists/17c85d5e7081f31dca5a2bf3917ef2f3
#owner: https://api.github.com/users/nickva

#!/usr/bin/env python

# ./just_view.py 500 1000

import sys
import time
import random
import requests

TIMEOUT=120
AUTH=('adm','pass')
URL='http://localhost:15984'
DBNAME = 'db'
Q = '8'

MAP_FUN = '''
function(doc){
    emit([doc.year, doc.month], doc.sales);
}
'''

def make_doc(counter) :
    return {
        '_id': str(counter),
        'year': random.randint(2001, 2025),
        'month': random.randint(1, 12),
        'sales': random.paretovariate(1)
    }

class Server:

    def __init__(self, url=URL, auth=AUTH, timeout=TIMEOUT):
        self.sess = requests.Session()
        self.sess.auth = auth
        self.url = url.rstrip('/')
        self.timeout = timeout

    def _apply_timeout(self, kw):
        if self.timeout is not None and 'timeout' not in kw:
            kw['timeout'] = self.timeout
        return kw

    def get(self, path = '', **kw):
        kw = self._apply_timeout(kw)
        r = self.sess.get(f'{self.url}/{path}', **kw)
        r.raise_for_status()
        return r.json()

    def post(self, path, **kw):
        kw = self._apply_timeout(kw)
        r = self.sess.post(f'{self.url}/{path}', **kw)
        r.raise_for_status()
        return r.json()

    def put(self, path, **kw):
        kw = self._apply_timeout(kw)
        r = self.sess.put(f'{self.url}/{path}', **kw)
        r.raise_for_status()
        return r.json()

    def delete(self, path, **kw):
        kw = self._apply_timeout(kw)
        r = self.sess.delete(f'{self.url}/{path}', **kw)
        r.raise_for_status()
        return r.json()

    def head(self, path, **kw):
        kw = self._apply_timeout(kw)
        r = self.sess.head(f'{self.url}/{path}', **kw)
        return r.status_code

    def version(self):
        return self.get()['version']

    def create_db(self, dbname, **kw):
        if dbname not in self:
            self.put(dbname, timeout=TIMEOUT, **kw)
        if dbname not in self:
            raise Exception(f"{dbname} could not be created")
        else:
            return True

    def bulk_docs(self, dbname, docs, timeout=TIMEOUT):
        return self.post(f'{dbname}/_bulk_docs', json = {'docs': docs})

    def bulk_get(self, dbname, docs, timeout=TIMEOUT):
        return self.post(f'{dbname}/_bulk_get', json = {'docs': docs})

    def compact(self, dbname, **kw):
        r = self.sess.post(f'{self.url}/{dbname}/_compact', json = {},  **kw)
        r.raise_for_status()
        return r.json()

    def config_set(self, section, key, val):
        url = f'_node/_local/_config/{section}/{key}'
        return self.put(url, data='"'+val+'"')

    def config_get(self, section, key):
        url = f'_node/_local/_config/{section}/{key}'
        return self.get(url)

    def __iter__(self):
        dbs = self.get('_all_dbs')
        return iter(dbs)

    def __str__(self):
        return "<Server:%s>" % self.url

    def __contains__(self, dbname):
        res = self.head(dbname)
        if res == 200:
            return True
        if res == 404:
            return False
        raise Exception(f"Unexpected head status code {res}")


def add_view(srv, db):
    srv.put(db + '/_design/d1', json = {
        "views": {
            "top1": {
               "map": MAP_FUN,
               "reduce": "_top1"
            },
            "top10": {
               "map": MAP_FUN,
               "reduce": "_top10"
            },
            "top100": {
               "map": MAP_FUN,
               "reduce": "_top100"
            },
            "bottom1": {
               "map": MAP_FUN,
               "reduce": "_bottom1"
            },
            "bottom10": {
               "map": MAP_FUN,
               "reduce": "_bottom10"
            },
            "bottom100": {
               "map": MAP_FUN,
               "reduce": "_bottom100"
            }
        },
        "autoupdate": False
    })


def main(n, b):
    dbname = DBNAME
    print("URL:",URL,"DB Name:",dbname)
    print("Batches: ",n,"Batchsize:",b,"Total:",n*b)
    s = Server(url = URL, timeout = TIMEOUT)
    if dbname in set(s): s.delete(dbname)
    s.create_db(dbname, params = {'q':Q})
    add_view(s, dbname)
    random.seed(42)
    for i in range(n):
       s.bulk_docs(dbname, [make_doc(i*b+j) for j in range(b)])
    print("View URL: ", f'{dbname}/_design/d1/_view/top10')


if __name__=='__main__':
    args=sys.argv[1:]
    if len(args)==2:
        n, b = int(args[0]), int(args[1])
        print(">>> n:", n, "b:", b, "total: ", n*b)
    else:
        n = 100
        b = 500
    main(n, b)
