#date: 2021-12-03T16:57:18Z
#url: https://api.github.com/gists/a4c7a0a2ac63e1f9ed0a05f9a56c620a
#owner: https://api.github.com/users/dgtlctzn

from time import time

responses: List[Dict] = []

start_time: float = time()

for url in URLS:
    responses.append(requests.get(url).json())

end_time: float = time()

print(end_time - start_time)