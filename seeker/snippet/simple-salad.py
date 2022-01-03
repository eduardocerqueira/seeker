#date: 2022-01-03T17:00:54Z
#url: https://api.github.com/gists/602ecccb8b47a8442f673aa8de555285
#owner: https://api.github.com/users/o018BUm8UQEEY2e5

from random import sample
words = "I wish I knew word salad, so I could say whatever without the consequence of someone understanding.".split()
' '.join(sample(words, k=len(words)))