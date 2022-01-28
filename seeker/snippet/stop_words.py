#date: 2022-01-28T17:03:50Z
#url: https://api.github.com/gists/4e0a0ad10ab988f794797d4d522eaf25
#owner: https://api.github.com/users/akshay-ravindran-96

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

text = "Enter the text where you want to remove the stop words like a, an, the, from your sentence"
tokens = word_tokenize(text)

sw_removed_ls = [term for term in tokens if not term in stopwords.words()]

print(sw_removed_ls)