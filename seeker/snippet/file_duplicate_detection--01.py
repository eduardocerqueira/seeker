#date: 2022-05-31T17:02:47Z
#url: https://api.github.com/gists/28eed8cc6f944d248ca7f9b9baa19e04
#owner: https://api.github.com/users/rahulremanan

class FileHash():
  def __init__(self, 
               chunk_size:int=4096, 
               crypto:str='blake2b')->None:
    self.chunk_size = chunk_size
    self.crypto = crypto
  def file_hash(self, fname:str)->str:
    _hash_fn = getattr(hashlib, self.crypto)()
    with open(fname, 'rb') as f:
      for _chunk in iter(lambda: f.read(self.chunk_size), b''):
        _hash_fn.update(_chunk)
    return _hash_fn.hexdigest()