#date: 2022-06-17T17:09:23Z
#url: https://api.github.com/gists/4c9d44b9fd7ace7b80bdaf7383de7477
#owner: https://api.github.com/users/parajain

import nltk
#nltk.download('omw-1.4')
import tqdm
from nltk.corpus import wordnet as wn

all_nouns = [word for synset in wn.all_synsets('n') for word in synset.lemma_names()]

inputphrase= 'Conversational Semantic Parsing Knowledge Graphs'
wordlens = [len(w) for w in inputphrase.split()]
t=0
cumm_wl = []
for wl in wordlens[:-1]:
    cumm_wl.append(wl + t)
    t = t + wl
cumm_wl.append(cumm_wl[-1]+1)

print(cumm_wl)
inputphrase = inputphrase.replace(' ', '').lower()
#inputphrase = inputphrase.split()
#chars = []
#for i in inputphrase:
#    chars.extend(list(i))

for n in tqdm.tqdm(all_nouns, total=len(all_nouns)):
    #if n != 'voltage': 
    #    continue
    #print('Now ****************************', n)
    idx = 0
    matched = []
    currn = inputphrase
    if '_' in n:
        continue
    chars = list(n)
    done = True
    coverage = cumm_wl.copy()
    last_covered = 0
    for c in chars:
        #print('curr ', currn)
        ci = currn.find(c)
        if ci > -1:
            #print(ci, currn[ci], c)
            matched.append(ci+idx)
            idx = idx + ci
            #print(idx, coverage, last_covered)
            
            if len(coverage) > 0 and idx <= coverage[0] and idx > last_covered:
                #print(last_covered)
                last_covered = coverage[0]
                coverage.pop(0)
            
            currn = inputphrase[idx:]
            #print(currn)
        else:
            done = False
            matched = []
            break
    
    if done and len(coverage) == 0: # remove coverage if you want to allow not using all words
        #print('************************* done ', n)
        s = ''
        for idx, cc in enumerate(list(inputphrase)):   
            if idx in matched:
                s += cc.upper()
            else:
                s += cc
        print(n, s, matched, coverage)

        #break


