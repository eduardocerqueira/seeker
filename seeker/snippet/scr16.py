#date: 2022-02-15T16:58:32Z
#url: https://api.github.com/gists/2ed1744db47295a2660afcf58ddf5569
#owner: https://api.github.com/users/vaibhavtmnit

from collections import Counter

text_data = ['i love pytorch', 'pytorch is the best', 'there are others but nothing as good as pytorch',
            'pytorch is future', 'viva la pytorch', 'pytorch rocks']
sentiment = [1]*len(text_data)

# Creating a quick vocab object
counter = Counter()
for text in text_data:
    
    counter.update(text.split(" "))
    
stoi={}
for i,(key,val) in enumerate(counter.items()):
    stoi[key] = i

    
# stoi   
# {'i': 0,
#  'love': 1,
#  'pytorch': 2,
#  'is': 3,
#  'the': 4,
#  'best': 5,
#  'there': 6,
#  'are': 7,
#  'others': 8,
#  'but': 9,
#  'nothing': 10,
#  'as': 11,
#  'good': 12,
#  'future': 13,
#  'viva': 14,
#  'la': 15,
#  'rocks': 16}