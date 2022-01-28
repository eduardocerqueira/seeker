#date: 2022-01-28T16:59:33Z
#url: https://api.github.com/gists/ed489cf4e86a82d773211862797cba3e
#owner: https://api.github.com/users/akshay-ravindran-96

import nltk
from nltk.tokenize import word_tokenize

text = word_tokenize("Text you want to Tag Parts of Speech")

nltk.pos_tag(text)