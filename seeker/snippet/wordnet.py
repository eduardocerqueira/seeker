#date: 2023-11-03T16:46:59Z
#url: https://api.github.com/gists/e0ee3aa991dabf9219d108ed955efe3a
#owner: https://api.github.com/users/ToroData

from nltk.corpus import wordnet

word = "security"
synsets = wordnet.synsets(word)
for synset in synsets:
    print(synset.definition())
