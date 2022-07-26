#date: 2022-07-26T17:08:33Z
#url: https://api.github.com/gists/b63bf5c708d0b2c5e697f3342ac8f159
#owner: https://api.github.com/users/thaisribeiro

import re
import requests
from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

async def get_summarization(summarization):
    formatted_text = summarization.text if summarization.url is None else scraping_text(summarization.url)
    stop_words = set(stopwords.words(str(summarization.language)))
    frequences_word = handle_frequence_word(stop_words, formatted_text)
    print('Dicionário de frequencia de palavras de parada', frequences_word)
    
def handle_frequence_word(stop_words, text):
    words = word_tokenize(text)
    frequences_word = dict()
    for word in words:
        if word in stop_words:
            continue
        if word in frequences_word:
            frequences_word[word] += 1
        else:
            frequences_word[word] = 1
            
    return frequences_word
    
def scraping_text(url):
    response = requests.get(url)
    soup = bs(response.content, 'lxml')
    paragraph = soup.find_all(name='p', attrs= {"class": "pw-post-body-paragraph"}, limit=40)
    joined_text = ''.join([p.text.strip().lower() for p in paragraph])
    joined_text = format_text(joined_text)
    return joined_text

def format_text(text):
    formatted_text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    formatted_text = re.sub(r'\[[0-9]*\]', ' ', formatted_text)
    formatted_text = re.sub(r'\s+', ' ', formatted_text)
    return formatted_text
