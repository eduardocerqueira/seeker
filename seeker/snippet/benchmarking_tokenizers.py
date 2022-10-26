#date: 2022-10-26T16:57:55Z
#url: https://api.github.com/gists/90bbb3307133e9fafc9a73709c3318c2
#owner: https://api.github.com/users/soldni

import subprocess
import sys

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "spacy",
    "blingfire",
    "tokenizers",
    "lorem"
])

import timeit
import lorem
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from blingfire import text_to_words
import tokenizers

hf_tok = "**********"
nlp = English()
spacy_tok = "**********"


 "**********"d "**********"e "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"_ "**********"s "**********"p "**********"a "**********"c "**********"y "**********"( "**********"t "**********"e "**********"x "**********"t "**********") "**********": "**********"
    return [t.text for t in spacy_tok(text)]

 "**********"d "**********"e "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"_ "**********"b "**********"l "**********"i "**********"n "**********"g "**********"f "**********"i "**********"r "**********"e "**********"( "**********"t "**********"e "**********"x "**********"t "**********") "**********": "**********"
    return text_to_words(text).split()

 "**********"d "**********"e "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"_ "**********"h "**********"f "**********"( "**********"t "**********"e "**********"x "**********"t "**********") "**********": "**********"
    return [e for e, _ in hf_tok.pre_tokenize_str(text)]


TIMES = 10_000
p = lorem.text()


start_time = timeit.default_timer()
for i in range(TIMES):
    tokenize_spacy(p)
print("Spacy: ", timeit.default_timer() - start_time)

start_time = timeit.default_timer()
for i in range(TIMES):
    tokenize_blingfire(p)
print("Blingfire: ", timeit.default_timer() - start_time)

start_time = timeit.default_timer()
for i in range(TIMES):
    tokenize_hf(p)
print("Huggingface: ", timeit.default_timer() - start_time)
