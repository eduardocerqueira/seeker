#date: 2024-02-21T17:02:17Z
#url: https://api.github.com/gists/2f181eaf8f10cbe209080713bf1a5324
#owner: https://api.github.com/users/mvandermeulen

# docker run -p 6379:6379 -it redis/redis-stack:latest

import concurrent.futures
from functools import lru_cache

import requests
from bs4 import BeautifulSoup
from pymemcache.client import base
from redis import Redis

url = 'https://free-proxy-list.net/'

redis = Redis(host='localhost', port=6379)
client = base.Client(('localhost', 11211))

if redis.exists('proxy_list'):
    proxy_list = redis.get('proxy_list').decode('utf-8').split('\n')
else:
    print("Fetching proxy list from the website...")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', attrs={'class': 'table table-striped table-bordered'})
    proxy_list = [f'{row.td.text.strip()}:{row.td.next_sibling.text.strip()}' for row in table.tbody.find_all('tr')]
    redis.set('proxy_list', '\n'.join(proxy_list))
    print("Proxy list fetched and stored in Redis.")

if redis.exists('successful_proxies'):
    successful_proxies = {k: v for k, v in redis.hgetall('successful_proxies').items() if float(v) < 2.0}
else:
    successful_proxies = {}

proxy_list = proxy_list[:100]
url = 'https://www.google.com/'
success_threshold = 5

@lru_cache(maxsize=None)
def test_proxy(proxy):
    try:
        print(f"Testing proxy: {proxy}")
        with requests.Session() as session:
            session.mount('http://', requests.adapters.HTTPAdapter(max_retries=3))
            session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
            response = session.get(url, proxies={'http': proxy, 'https': proxy}, timeout=10)
        if response.status_code == 200:
            response_time = response.elapsed.total_seconds()
            successful_proxies[proxy] = response_time
            print(f"Proxy {proxy} is working. Response time: {response_time:.2f} seconds")
            return True
        else:
            print(f"Proxy {proxy} returned status code {response.status_code}")
    except Exception as e:
        print(f"Error occurred while testing proxy {proxy}: {str(e)}")
    return False

with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(test_proxy, proxy) for proxy in proxy_list]
    success_count = 0
    for future in concurrent.futures.as_completed(futures):
        if future.result():
            success_count += 1
            if success_count >= success_threshold:
                break

for proxy, response_time in successful_proxies.items():
    redis.hset('successful_proxies', proxy, response_time)
    client.set(proxy, str(response_time), expire=60*60*24*7)

top_n = 3
proxy_list = redis.hgetall('successful_proxies')
sorted_proxies = sorted(proxy_list.items(), key=lambda x: float(x[1]))
print(f"Top {top_n} Fastest Proxies:")
for i, (proxy, response_time) in enumerate(sorted_proxies[:top_n]):
    print(f"{i+1}. Proxy: {proxy}, Response Time: {response_time:.2f} seconds")