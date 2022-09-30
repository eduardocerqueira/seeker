#date: 2022-09-30T17:05:13Z
#url: https://api.github.com/gists/a384181ef3ab74afe6414fb916ae7e47
#owner: https://api.github.com/users/pradhyumna85

# %% imports
from math import log

# %% getting words
# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
with open("words-by-frequency.txt") as f:
    words = f.read().split()

wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
maxword = max(len(x) for x in words)

# %% function defination

def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return " ".join(reversed(out))

texts = [
     'interestedinducedand',
    'thumbgreenappleactiveassignmentweeklymetaphor',
    "thereismassesoftextinformationofpeoplescommentswhichisparsedfromhtmlbuttherearen odelimitedcharactersinthemforexamplethumbgreenappleactiveassignmentweeklymetapho rapparentlytherearethumbgreenappleetcinthestringialsohavealargedictionarytoquery whetherthewordisreasonablesowhatsthefastestwayofextractionthxalot.",
    "itwasadarkandstormynighttherainfellintorrentsexceptatoccasionalintervalswhenitwascheckedbyaviolentgustofwindwhichsweptupthestreetsforitisinlondonthatoursceneliesrattlingalongthehousetopsandfiercelyagitatingthescantyflameofthelampsthatstruggledagainstthedarkness.",
]

# %% manual function
print('Manual infer spaces function\n')
for text in texts:
    print(f'############## Original:')
    print(text)
    print(f'\n############## Processed:')
    print(infer_spaces(text))
    print('\n--------------------------------------------\n')

# %% using wordninja
import wordninja

print('Wordninja package implementation\n')
for text in texts:
    print(f'############## Original:')
    print(text)
    splitted = wordninja.split(text)
    print(f'\n############## Processed:')
    print(' '.join(splitted))
    print('\n--------------------------------------------\n')




# %% Credits

'''
https://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words
words: http://tinypaste.com/c1666a6b

package inspired from this stack: https://github.com/keredson/wordninja, https://pypi.org/project/wordninja/, pip install wordninja
'''