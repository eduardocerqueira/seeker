#date: 2022-01-21T17:14:05Z
#url: https://api.github.com/gists/402161dc13f588b6b42eea0f9b3899a7
#owner: https://api.github.com/users/zemanlx

#!/bin/bash

# Install American dictionary
sudo apt-get install wamerican

# See first 10 words
head /usr/share/dict/american-english

# Count words in the dictionary
wc -l /usr/share/dict/american-english

# Create Wordle dictionary (all five-letter words)
grep -P "^[a-z]{5}$" /usr/share/dict/american-english \
    > wordle.dictionary

# Count words in the Wordle dictionary
wc -l wordle.dictionary

# Get execution time of Wordle dictionary creation
time grep -P "^[a-z]{5}$" /usr/share/dict/american-english \
    > wordle.dictionary