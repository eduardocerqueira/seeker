#date: 2022-04-29T17:13:34Z
#url: https://api.github.com/gists/f995396b312cf3aac92e14d85e4eb9ae
#owner: https://api.github.com/users/tcyrus

import struct
import numpy
import tqdm
import random
import bz2

#dataset_file = 'GoogleNews-vectors-negative300.bin'
dataset_file = 'dataset-trimmed.bin.bz2'
word_filter = lambda z : z.isalpha() and z.lower() == z

# open dataset file and read first line
f = bz2.BZ2File(dataset_file)
#f = open(dataset_file,'rb')
line = ''
while True:
    c = f.read(1).decode()
    assert len(c) == 1
    line += c
    if c == '\n':
        break

numwords,numdims = map(int,line.split()) # first line

def get_word(): # read a word in the dataset file
    word = b''
    while True:
        c = f.read(1)
        assert len(c) == 1
        word += c
        if c == b' ':
            return word[:-1].decode()

word2vec = dict() # map word to numpy vectors
print('Loading dataset...')
for _ in tqdm.tqdm(range(numwords)):
    word = get_word()
    floatbin = f.read(numdims*4) # little endian
    if word_filter(word):
        floats = [struct.unpack('<f',floatbin[4*i:4*i+4])[0]
                for i in range(numdims)]
        word2vec[word] = numpy.array(floats,dtype=float)

assert f.read(1) == b''
f.close()

# cache vector norms
word2norm = dict()
print('Caching vector norms...')
for word in tqdm.tqdm(word2vec):
    norm = numpy.linalg.norm(word2vec[word])
    word2norm[word] = norm

# find words in a given similarity range, limit = -1 for no limit
def near_words(words,word,sim_lo=0.5,sim_hi=1.0,limit=100,progress=True):
    assert -1.0 <= sim_lo <= sim_hi <= 1.0
    result = []
    if progress:
        print('Computing near words...')
        iter_obj = tqdm.tqdm(words)
    else:
        iter_obj = words
    for w in iter_obj:
        dot = numpy.dot(word2vec[word],word2vec[w])
        sim = dot/(word2norm[word]*word2norm[w])
        if sim_lo <= sim <= sim_hi:
            result.append((w,sim))
            if len(result) == limit:
                break
    return result

# print near words
def show_near_words(word,sim_lo=0.5,sim_hi=1.0,limit=100):
    words = near_words(word,sim_hi,sim_lo,limit)
    words = sorted(words,key=lambda x:-x[1])
    for w in words:
        print('%10.6f %s'%(w[1],w[0]))

# compute intersection with multiple word data
# input list of (word,sim) pairs, use sim as displayed on semantle
def solve_semantle(data):
    words = set(word2vec.keys())
    is_first = True
    for word,sim in data:
        sim = sim/100.0
        if is_first:
            print('Finding words for first intersection...')
        near_data = near_words(words,word,sim-0.001,sim+0.001,-1,is_first)
        is_first = False
        words &= set(n[0] for n in near_data)
    return words

