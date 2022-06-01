#date: 2022-06-01T17:03:53Z
#url: https://api.github.com/gists/c08f1048609ecdc0e6f3cb253eb5510e
#owner: https://api.github.com/users/peterbe

import statistics
import time
import requests
from collections import Counter, defaultdict

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


def requests_retry_session(
    retries=2,
    backoff_factor=0.2,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def fmt(s):
    if s > 60:
        return f"{s / 60:.1f}min"
    if s > 10:
        return f"{s:.1f}s"
    return f"{s * 1000:.1f}ms"


counter = Counter()
iterations = 1
# while r.status_code == 200:
prev = {}
elapses_by_x_cache = defaultdict(list)

def summarize():
    print("Times by X-Cache value...")
    for value, times in elapses_by_x_cache.items():
        print("\tHeader:", repr(value).ljust(35), "#", len(times), "MEAN:", fmt(statistics.mean(times)), 'MEDIAN:', fmt(statistics.median(times)))
    print('\n')

timeline = None
session = requests_retry_session()

while iterations < 500:
    # print(r.headers)
    # r = session.get('http://staging.ghdocs.com/en')
    r = session.get('https://www.peterbe.com/')
    # r = session.get('https://developer.mozilla.org/en-US/')
    # r = requests.get('http://localhost:4000/en')
    counter[r.status_code] += 1
    if r.status_code == 200:


        x_cache = r.headers.get('x-cache', '*none*')
        key = f"{x_cache}:{r.headers.get('cache-control', '*none*')}"
        if timeline is None:
            timeline = [
                (key, time.time())
            ]
        elif timeline[-1][0] != key:
            timeline.append(
                (key, time.time())
            )

        elapses_by_x_cache[key].append(r.elapsed.total_seconds())
        print(
            fmt(r.elapsed.total_seconds()).ljust(8),
            f"#{iterations}".ljust(5),
        )
        # for k in ('x-cache', 'x-middleware-cache', 'cache-control', 'set-cookie'):
        for k in ('x-cache',  'cache-control', 'set-cookie'):
            v = r.headers.get(k)
            if k != 'set-cookie' and k in prev and prev[k] != v:
                print(k, "CHANGED! FROM", prev[k], "TO", v)
                print()
                # time.sleep(2)
            print('\t', k.ljust(20), r.headers.get(k))
            prev[k] = v

        if 'hit' in r.headers.get('x-cache', '').lower():
            time.sleep(0.1)
    else:
        print(r.status_code, ":(")
        time.sleep(1)

    iterations += 1
    # if not iterations % 100:
    #     summarize()

summarize()
# print(r.status_code)


previous = None
for value, t in timeline:
    if previous:
        print(f"FROM {repr(previous[0]):<35} TO {repr(value):<35}  {fmt(t - previous[1])}")
    previous = (value, t)
