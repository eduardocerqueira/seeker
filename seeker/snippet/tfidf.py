#date: 2022-09-16T17:30:54Z
#url: https://api.github.com/gists/f19b451af55a9541a1ac016a24e32981
#owner: https://api.github.com/users/jacobobryant

"""
Usage: python tfidf.py extract_keywords

Reads from storage/keywords/corpus.csv, which has columns `<ID>,<Title>,<Description>`. Writes
keywords to storage/keywords/output.csv

I use this for newsletter topic modeling at https://thesample.ai/.
"""
import sys
import nltk
import re
import csv
import random
import numpy as np
import json
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import fasttext

exclusions = [
        'subscribe',
        'newsletter',
        'weekly',
        'things',
        'daily',
        'day',
        'ideas',
        'week',
        'inbox',
        'delivered',
        'interesting',
        'better',
        'best',
        'thoughts',
        'latest',
        'new',
        'analysis',
        'good',
        'people'
        'free',
        'like',
        'notes'
        ]

def extract_keywords():
    try:
        nltk.corpus.stopwords.words('english')
    except:
        nltk.download('stopwords')

    with open('storage/keywords/corpus.csv', newline='') as csvfile:
        nls = [{'id': row[0], 'text': row[1] + ' ' + row[2]} for row in csv.reader(csvfile)]
    random.shuffle(nls)

    tfidf_vectorizer = TfidfVectorizer(
        use_idf=True,
        max_df=0.8,
        min_df=3/len(nls),
        stop_words='english'
    )

    texts = [nl['text'] for nl in nls]
    matrix = tfidf_vectorizer.fit_transform(texts)

    with open('storage/keywords/output.csv.tmp', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for nl, v in list(zip(nls, matrix)):
            #print(nl['text'])
            x = list(zip(v.data, v.indices))
            x.sort(key=lambda pair: -pair[0])
            feature_names = tfidf_vectorizer.get_feature_names()
            keywords = [feature_names[i]
                        for value, i in x
                        if feature_names[i] not in exclusions]
            #print(', '.join(keywords[:5]))
            #print()
            writer.writerow([nl['id']] + keywords[:5])
    shutil.copyfile('storage/keywords/output.csv.tmp', 'storage/keywords/output.csv')

def main():
    eval(sys.argv[1])(*sys.argv[2:])

if __name__ == "__main__":
    main()
