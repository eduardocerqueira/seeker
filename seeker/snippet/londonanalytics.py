#date: 2022-03-22T17:10:45Z
#url: https://api.github.com/gists/3e433f2ebae66cf42f7a19cb1a8c0c4b
#owner: https://api.github.com/users/abhinay

import requests

text = 'My email address is spy@ninja.com and my number is 020 3477 9393 so call me maybe?'

response = requests.post(
    'https://www.londonanalytics.co.uk/api/named_entities/find?api_key=YOUR-API-KEY',
    json={'text': text}
)

entities = response.json()

for entity in entities:
    print(entity['type'] +": "+ entity['value'])
