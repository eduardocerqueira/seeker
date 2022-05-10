#date: 2022-05-10T16:57:37Z
#url: https://api.github.com/gists/2585ce4bd6da7c78de3928b9d01e07ac
#owner: https://api.github.com/users/sergioburdisso

from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def wordpiece_tokenizer(text):
    return bert_tokenizer.convert_ids_to_tokens(bert_tokenizer(text)['input_ids'])[1:-1]

tfidf_vectorizer = TfidfVectorizer(tokenizer=wordpiece_tokenizer)  # add other params as well, e.g. ngram_range

tfidf_vectorizer.fit(["this is a wordpiece test", "esto es una pruebaza"])

print(tfidf_vectorizer.vocabulary_)
# {'this': 8, 'is': 5, 'a': 2, 'word': 10, '##piece': 0, 'test': 7, 'esto': 4, 'es': 3, 'una': 9, 'prueba': 6, '##za': 1}
