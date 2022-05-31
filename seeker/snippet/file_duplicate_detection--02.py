#date: 2022-05-31T17:04:15Z
#url: https://api.github.com/gists/492201316153622adec07715cdd8a0e4
#owner: https://api.github.com/users/rahulremanan

class FileDedup(FileHash):
  def __init__(self,
               crypto:str='blake2b', 
               chunk_size:int=2048):
    super().__init__()
    self.crypto = crypto
    self.chunk_size = chunk_size
  def __call__(self,
               file_list:list)->dict:
    file_compare = {}
    for f in tqdm(file_list):
      try:
        file_compare[self.file_hash(f)].append(f)
      except KeyError:    
        file_compare[self.file_hash(f)] = [f]
    return file_compare