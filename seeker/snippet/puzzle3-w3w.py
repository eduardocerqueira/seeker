#date: 2023-05-29T16:57:05Z
#url: https://api.github.com/gists/fa252eacb4f356b0b2052408d605e344
#owner: https://api.github.com/users/schrobby

#!/usr/bin/env python

import requests
import time
import sys
import csv
from itertools import permutations
from flatten_dict import flatten

api_key = 'API-KEY'


# read word list and create permutations list
filename = sys.argv[1] if len(sys.argv) > 1 else 'words.txt'
with open(filename, 'r') as file:
    words = file.readline().split(',')
    words = [word.strip().lower() for word in words]

permutations_list = ['.'.join(permutation) for permutation in permutations(words, 3)]

print('Using words: ' + ', '.join(words))
print(f'Running through {len(permutations_list)} possible permutations:')


# scrape w3w data and store in list
response_list = []
for item in permutations_list:
    url = f'https://api.what3words.com/v3/convert-to-coordinates?words={item}&key={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        response_list.append(data)
        print(item)
    else:
        print(f'Error occurred for {item}. Status code: {response.status_code}')
    time.sleep(0.1)


# convert dict to csv and save in file
json_data = [flatten(x, reducer='underscore') for x in response_list]
column_keys = json_data[0].keys()

filename = sys.argv[2] if len(sys.argv) > 2 else 'w3w.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=column_keys)
    writer.writeheader()  
    writer.writerows(json_data)

print(f'W3W data saved to {filename}')
