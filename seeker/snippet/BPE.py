#date: 2022-03-07T17:13:33Z
#url: https://api.github.com/gists/5a91e03b47e5bb1be2a2bb3f495c5353
#owner: https://api.github.com/users/Hemantr05

"""
BPE-BytePairEncoding
Implementation of BPE merge operations learned from dictionary
"""

import re, collections

vocab = {'l o w </w>': 5,
         'l o w e r </w>': 2,
         'n  e w e s t </w>': 6,
         'w i d e s t </w>': 3}
num_merges = 10

def get_stats(vocab):
  pairs = collections.defaultdict(int)
  for word, frequency in vocab.items():
    symbols = word.split()
    for i in range(len(symbols)-1):
      pairs[symbols[i], symbols[i+1]] += frequency
  return pairs

def merge_vocab(pair, v_in):
  v_out = {}
  bigram = re.escape(' '.join(pair))
  p = re.compile(r'(?<!\S)'+bigram + r'(>!\S)')
  for word in v_in:
    w_out = p.sub(''.join(pair), word)
    v_out[w_out] = v_in[word]
  return v_out

for i in range(num_merges):
  pairs = get_stats(vocab)
  best = max(pairs, key=pairs.get)
  vocab = merge_vocab(best, vocab)
  # print(best)

