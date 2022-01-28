#date: 2022-01-28T16:52:02Z
#url: https://api.github.com/gists/203e0fdb611e9e0b185ec12c8e169c40
#owner: https://api.github.com/users/akshay-ravindran-96

# NLTK
import nltk
nltk.download('punkt')
text = "write text to convert into tokens"

sentences = nltk.sent_tokenize(text)

words = nltk.word_tokenize(text)