#date: 2023-01-09T17:12:26Z
#url: https://api.github.com/gists/92859b32f6281ccd582ec51df6b6d0de
#owner: https://api.github.com/users/cmdoret

import sys
from transformers import pipeline

# Manually picked broadest terms from SRAO https://www.ebi.ac.uk/ols/ontologies/srao
candidate_labels = [
    "humanities",
    "social sciences",
    "chemistry",
    "earth science",
    "life science",
    "mathematics",
    "physics",
    "civil engineering",
    "bioengineering",
    "computer science",
    "electrical engineering",
    "energy engineering",
    "industrial engineering",
    "systems engineering",
]

with open(sys.argv[1], "r") as readme:
    text = readme.read()[:1023]

summary_model = 'sshleifer/distilbart-cnn-12-6'
summarizer = pipeline('summarization', model=summary_model, min_length=10, max_length=100)
summary = summarizer(text)[0]['summary_text']
print(f"Summary: {summary}")

keywords_model = "facebook/bart-large-mnli"
classifier = pipeline("zero-shot-classification", model=keywords_model)
kws = classifier(text, candidate_labels, multi_label=True)
matches = [kw for kw, score in zip(kws['labels'], kws['scores']) if score > 0.3]
print(f"Keywords: {matches}")