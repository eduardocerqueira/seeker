#date: 2022-02-25T16:49:03Z
#url: https://api.github.com/gists/8c05d56b6f231e391fa610cd0af0451b
#owner: https://api.github.com/users/viz-prakash

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.poolmanager import PoolManager
import requests


class SourcePortAdapter(HTTPAdapter):
    """"Transport adapter" that allows us to set the source port."""
    def __init__(self, port, *args, **kwargs):
        self._source_port = port
        super(SourcePortAdapter, self).__init__(*args, **kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = PoolManager(
            num_pools=connections, maxsize=maxsize,
            block=block, source_address=('', self._source_port))
        
s = requests.session()
s.mount('http://127.0.0.1:8080', SourcePortAdapter(54321))
s.get('http://127.0.0.1:8080/my_uri?with=args')