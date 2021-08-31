#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

from pymemcache.client.base import Client


def set(key,value):
    client = Client(('localhost', 11211))
    client.set(key, value)

def get(key):
    client = Client(('localhost', 11211))
    return client.get(key)
