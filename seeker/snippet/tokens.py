#date: 2022-02-01T16:56:53Z
#url: https://api.github.com/gists/580362931b024350b2af8bf76715eef3
#owner: https://api.github.com/users/priyanka27s

# Author: Priyanka Sinha, Date: 1st February 2022
# install spacy and download the 'en_core_web_sm' model
# run on command line using python3 tokens.py <input-text-file> <stopwords-file> <integer n>
# Example: python3 tokens.py alice_in_wonderland.txt 1-1000.txt 5
# Output: 
# Count	Word
# ===	====

# 453 	 said
# 395 	 Alice
# 124 	 little
# 84 	 like
# 83 	 went


import sys 
import spacy

with open(sys.argv[1],'r') as fp:
	alltext = fp.read()

with open(sys.argv[2],'r') as ofp:
	stopwords = ofp.readlines()

n = int(sys.argv[3])

nlp = spacy.load('en_core_web_sm')

nlp.Defaults.stop_words = stopwords

doc = nlp(alltext)

counters = {}

for token in doc:
	w = token.text
	if (not token.is_alpha) or token.is_stop:
		continue
	if w in counters:
		counters[w] += 1
	else:
		counters[w] = 1


print("Count\tWord\n===\t====\n")

for w in sorted(counters, key=counters.get, reverse=True):
	print(counters[w],"\t",w)
	n -= 1
	if (n <= 0):
		break
