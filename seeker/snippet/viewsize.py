#date: 2022-12-07T16:50:05Z
#url: https://api.github.com/gists/f4177e581a5ba275b854b9a5af875d84
#owner: https://api.github.com/users/nickva

#!/usr/bin/env python

# still python2, sorry, old script
# pip install CouchDB
#  ./viewsize.py
# file      active    external  dtavg
# 18080120  17607050  14984706  45.973

import argparse
import sys
import couchdb
import random
import string
import uuid
import time
import copy
import itertools

URL = 'http://adm:pass@127.0.0.1:15984'
DBNAME = 'thedb'
TEMPLATE =  '%-9s %-9s %-9s %-9s'
DEFAULT_MAP = 'function(d){emit("K"+d._id, d.v);}'

REDUCE_FUN = """
function(key, values, rereduce) {
  var result = {count: 0};
  for(i=0; i < values.length; i++) {
    if(rereduce) {
        result.count = result.count + values[i].count;
    } else {
        result.count = values.length;
    }
  }
  return(result);
}
"""

PARAMS = [
    ('num',          'n', 500000,     "Number of documents"),
    ('batch_size',   'b', 2500,        "Batch size"),
    ('size',         's', 10,          "Emit value size"),
    ('query',        'q', False,       "Query after every batch?"),
    ('min_key_size', 'k', 1,           "Minimum key size"),
    ('random_keys',  'x', False,       "Use random keys?"),
    ('random_seed',  'X', 4,           "Random seed"),
    ('map_fun',      'm', DEFAULT_MAP, "Map function"),
    ('reduce_fun',   'r', REDUCE_FUN,  "Reduce function")
]

def add_view(db, args):
    if args.reduce_fun:
        view = {'reduce': args.reduce_fun}
    else:
        view = {}
    view['map'] = args.map_fun
    db['_design/des1'] = {"views": {"v1": view}, "autoupdate":False}

def main(args):
    param_names = [pn for (pn, _, _, _) in PARAMS]
    param_values = [None for _ in xrange(len(param_names))]
    is_default = set()
    default_values = {}
    for pname, _, val, _ in PARAMS:
        default_values[pname] = val
    for an, av in args._get_kwargs():
        if isinstance(av, list):
            if av == []:
                av = [default_values[an]]
                is_default.add(an)
            param_values[param_names.index(an)] = av
    print TEMPLATE % ("file", "active", "external", "dtavg")
    for vtup in itertools.product(*param_values):
        zipped = zip(param_names, vtup)
        paramstr = ",".join(["%s=%s" % (n, v) for (n, v) in zipped
            if n not in is_default])
        run_args = copy.copy(args)
        for (n, v) in zipped:
            setattr(run_args, n, v)
        run(run_args, paramstr)
    print

def run(args, paramstr):
    random.seed(args.random_seed)
    s = couchdb.Server(args.url)
    version = s.version()
    dbname = "%s-%s" % (args.dbname, uuid.uuid4().hex)
    if dbname in s:
        s.delete(dbname)
        time.sleep(args.view_query_timeout)
    reqtimes = []
    db = s.create(dbname)
    add_view(db, args)
    n = args.num
    b = args.batch_size
    for i in xrange(n / b):
       db.update([_doc(i * b + j, args) for j in xrange(b)])
       if args.query:
           t0 = time.time()
           len(db.view('des1/v1'))
           reqtimes.append(time.time() - t0)
    stopped_at = n - n % b
    db.update([_doc(i, args) for i in xrange(stopped_at, n)])
    t0 = time.time()
    len(db.view('des1/v1'))
    reqtimes.append(time.time() - t0)
    dtavg = sum(reqtimes) / len(reqtimes)
    time.sleep(args.view_query_timeout)
    sizes = db.info('des1')['view_index']['sizes']
    fsize, asize, esize = sizes['file'], sizes['active'], sizes['external']
    print TEMPLATE % (fsize, asize, esize, "%.3f" % dtavg), paramstr
    sys.stdout.flush()
    #s.delete(dbname)

def _doc(i, args):
   return {'_id': _id(i, args), 'v': _data(args)}

def _data(args):
    if args.alphabet:
        alphabet = args.alphabet
    else:
        alphabet = string.ascii_letters + string.digits
    return ''.join(random.choice(alphabet) for _ in xrange(args.size))

def _id(i, args):
    if args.random_keys:
        i = random.randint(0, 999999)
    key = '%06d' % i
    extend = args.min_key_size - len(key)
    if extend > 0:
        key = key + 'x' * extend
    return key

def _str2bool(val):
    val = val.lower()
    if val in ['true', 't', 'yes', 'yep']:
        return True
    else:
        return False

def _args():
    description = "Make a view, add docs and measure size"
    p = argparse.ArgumentParser(description = description)
    p.add_argument('-u', '--url', default=URL, help = "Server URL")
    p.add_argument('-d', '--dbname', default=DBNAME, help = "DB name")
    p.add_argument('-a', '--alphabet', default=None, help = "Data alphabet")
    p.add_argument('-t', '--view-query-timeout', type=int, default=10,
                   help = "Hold-off used by db to commit view changes")
    for pname, short, default, hstr in PARAMS:
        atype = type(default)
        ashort = '-' + short
        along = '--' + pname
        if atype is bool:
            atype = _str2bool
        p.add_argument(ashort, along, type=atype, action="append",
            default=[], help=hstr)
    return p.parse_args()

if __name__=='__main__':
    main(_args())
