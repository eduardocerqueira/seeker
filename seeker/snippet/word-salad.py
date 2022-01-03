#date: 2022-01-03T17:06:44Z
#url: https://api.github.com/gists/478d56f8093759c38a7f28bb0eb385ae
#owner: https://api.github.com/users/o018BUm8UQEEY2e5

from functools import reduce
from random import sample
from nltk import pos_tag, word_tokenize

def word_salad(sentence):
    #TODO: preserve relative ordering of parens
    #      don't interleave different quote types
    #      nltk leaves quotes attached to words
    class InvalidCombination(Exception):
        pass
    def combinations(a, b):
        fs = frozenset
        if (a[1] != '.' and b[1] != '.'):
            return (' '.join((a[0], b[0])), b[1])
        if a[1] == '.' and b[1] != '.':
            if a[0] in fs('({[/\'"'):
                return (''.join((a[0], b[0])), b[1])
            return (' '.join((a[0], b[0])), b[1])
        if a[1] != '.' and b[1] == '.':
            if b[0] in fs('({['):
                return (' '.join((a[0], b[0])), b[1])
            return (''.join((a[0], b[0])), b[1])
        #if a[1] == '.' and b[1] == '.':
        if (a[0] in fs('([{') and b[0] in fs('"\'')) or \
           (a[0] in fs('!?') and b[0] in fs('!?')) or \
           (a[0] in fs(')]}') and b[0] in fs('.,;?!')) or \
           (a[0] in fs('.?!') and b[0] in fs(')]}')):
            return (''.join((a[0], b[0])), b[1])
        raise InvalidCombination
    # inefficient but not in developer time
    while True:
        try:
            tagged_words = pos_tag(word_tokenize(sentence), tagset='universal')
            return reduce(combinations, sample(tagged_words, k=len(tagged_words)))[0]
        except InvalidCombination:
            continue

print(word_salad("I wish I knew word salad, so I could say whatever without the consequence of someone understanding."))