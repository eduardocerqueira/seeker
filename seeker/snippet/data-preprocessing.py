#date: 2024-09-17T16:59:16Z
#url: https://api.github.com/gists/af16acf0a46155283562cb074498248c
#owner: https://api.github.com/users/docsallover

from nltk.corpus import stopwords
nltk.download('stopwords')  # Download stopwords if not already downloaded

# Remove the date column (assuming it's not relevant for analysis)
data.drop(["date"], axis=1, inplace=True)
print(data.head())

# Remove the title column (focusing on text content)
data.drop(["title"], axis=1, inplace=True)
print(data.head())

# Convert text to lowercase for consistency
data['text'] = data['text'].apply(lambda x: x.lower())
print(data.head())

import string

# Remove punctuation for cleaner analysis
def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str

data['text'] = data['text'].apply(punctuation_removal)
print(data.head())

# Remove stopwords for better word representation
stop = stopwords.words('english')

data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
print(data.head())