#date: 2022-04-29T17:13:34Z
#url: https://api.github.com/gists/f995396b312cf3aac92e14d85e4eb9ae
#owner: https://api.github.com/users/tcyrus

import struct
import numpy
import tqdm
import random

dataset_file = 'GoogleNews-vectors-negative300.bin'
output_file = 'dataset-trimmed.bin'
word_filter = lambda z : z.isalpha() and z.lower() == z

outf = open(output_file,'wb')

# open dataset file and read first line
f = open(dataset_file,'rb')
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
            return word

word2vec = dict() # map word to numpy vectors
print('Loading dataset...')
for _ in tqdm.tqdm(range(numwords)):
    word = get_word()
    floatbin = f.read(numdims*4) # little endian
    if word_filter(word.decode()[:-1]):
        word2vec[word] = floatbin

assert f.read(1) == b''
f.close()

outf.write(('%d %d\n'%(len(word2vec),numdims)).encode())
print('Writing trimmed dataset...')
for word in tqdm.tqdm(word2vec):
    outf.write(word)
    outf.write(word2vec[word])

outf.close()
