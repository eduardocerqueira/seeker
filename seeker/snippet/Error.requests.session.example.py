#date: 2024-02-02T17:00:28Z
#url: https://api.github.com/gists/8dfb878a8ca2ed3d5f42853f75612c6f
#owner: https://api.github.com/users/furyutei

# -*- coding: utf-8 -*-

import requests

test_url = 'https://*****' # 【テストしたいURLを入れる】

session = requests.session()

print(f'test_url: {test_url}')
response = session.get(test_url)
#requests.exceptions.SSLError: HTTPSConnectionPool(host='*****', port=443): Max retries exceeded with url: / (Caused by SSLError(SSLError(1, '[SSL: DH_KEY_TOO_SMALL] dh key too small (_ssl.c:992)')))

print(f'status_code: {response.status_code}')
