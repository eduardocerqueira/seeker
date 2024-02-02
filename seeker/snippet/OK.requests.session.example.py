#date: 2024-02-02T17:00:28Z
#url: https://api.github.com/gists/8dfb878a8ca2ed3d5f42853f75612c6f
#owner: https://api.github.com/users/furyutei

# -*- coding: utf-8 -*-

import requests
import re

test_url = 'https://*****' # 【テストしたいURLを入れる】

def get_prefix(target_url): #{
  from urllib.parse import urlparse
  parts = urlparse(target_url)
  return f'{parts.scheme}://{parts.hostname}'
#}

def mount_session_adapter(
  session,
  prefix,
  custom_cipher_list # [SSLCipherSuite Directive](https://httpd.apache.org/docs/2.4/mod/mod_ssl.html#sslciphersuite)に上がっているようなTag(Cipher Tag/Protocol)のリスト
): #{
  """
  sessionでリソースを取得する際、prefixにマッチするURLについては、
  カスタマイズされた(custom_cipher_listを設定したSSL-Contextが適用された)アダプタを用いるように設定
  
  参考: [ssl - How to alter cipher suite used with Python requests? - Stack Overflow](https://stackoverflow.com/questions/77262501/how-to-alter-cipher-suite-used-with-python-requests/77270120#77270120)
  """
  custom_ciphers = ':'.join(custom_cipher_list)
  
  class CustomCipherAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
      kwargs['ssl_context'] = requests.packages.urllib3.util.ssl_.create_urllib3_context(ciphers=custom_ciphers)
      return super(CustomCipherAdapter, self).init_poolmanager(*args, **kwargs)
  
  session.mount(prefix, CustomCipherAdapter())
  """
   |  mount(self, prefix, adapter)
   |      Registers a connection adapter to a prefix.
   |
   |      Adapters are sorted in descending order by prefix length.
  """
#}

session = requests.session()

prefix = get_prefix(test_url)
mount_session_adapter(session, prefix, ['HIGH', '!DH', '!aNULL',])

print(f'test_url: {test_url}')
response = session.get(test_url)

print(f'status_code: {response.status_code}')
