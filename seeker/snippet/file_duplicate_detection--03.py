#date: 2022-05-31T17:23:05Z
#url: https://api.github.com/gists/e8863e072f43ecf0c98b373fa7709711
#owner: https://api.github.com/users/rahulremanan

def find_duplicates(dedup_dict):
  num_dup = 0  
  for i, k in tqdm(enumerate(dedup_dict)):
    if len(dedup_dict[k])>1:
      print('\n', dedup_dict[k], '\n ')
      num_dup += 1
  print(f'Number of files with duplicates: {num_dup}')
  
find_duplicates(dedup_dict)