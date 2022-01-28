#date: 2022-01-28T16:56:02Z
#url: https://api.github.com/gists/b24687b3694d3a36ccefcef0ef5f6151
#owner: https://api.github.com/users/akshay-ravindran-96

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

sm = PorterStemmer()
text = ""

words = word_tokenize(text)
for w in words:
    print(sm.stem(w))