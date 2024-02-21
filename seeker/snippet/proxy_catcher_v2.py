#date: 2024-02-21T17:02:17Z
#url: https://api.github.com/gists/2f181eaf8f10cbe209080713bf1a5324
#owner: https://api.github.com/users/mvandermeulen

# Try this if you don't have Redis installed, and want a simpler version (it does the same thing)
import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
from functools import lru_cache

url = 'https://free-proxy-list.net/'


@lru_cache(maxsize=None)
def get_proxy_list():
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', attrs={'class': 'table table-striped table-bordered'})
        return [
            f'{row.td.text.strip()}:{row.td.next_sibling.text.strip()}'
            for row in table.tbody.find_all('tr')
        ]
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"Error: {e}")
        return []


@lru_cache(maxsize=None)
def test_proxy(proxy):
    try:
        with requests.Session() as session:
            session.mount('http://', requests.adapters.HTTPAdapter(max_retries=3))
            session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
            response = session.get('https://www.google.com/', proxies={'http': proxy, 'https': proxy}, timeout=10)
        if response.status_code == 200:
            return response.elapsed.total_seconds()
    except requests.exceptions.Timeout:
        print(f'Timeout error for proxy {proxy}')
    except requests.exceptions.ProxyError:
        print(f'Proxy error for proxy {proxy}')
    except requests.exceptions.ConnectionError:
        print(f'Connection error for proxy {proxy}')
    except requests.exceptions.RequestException as e:
        print(f'Error for proxy {proxy}: {e}')
    return None


proxy_list = get_proxy_list()[:100]
success_threshold = 5
successful_proxies = {}

for proxy in proxy_list:
    response_time = test_proxy(proxy)
    if response_time is not None:
        successful_proxies[proxy] = response_time
        if len(successful_proxies) >= success_threshold:
            break

sorted_proxies = sorted(successful_proxies.items(), key=lambda x: float(x[1]))

print("Top 3 Fastest Proxies:")
for i, (proxy, response_time) in enumerate(sorted_proxies[:3]):
    print(f"{i + 1}. Proxy: {proxy}, Response Time: {response_time:.2f} seconds")
