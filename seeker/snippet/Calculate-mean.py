#date: 2021-10-26T17:04:53Z
#url: https://api.github.com/gists/e7602e25cd25dec297df314d802e8be3
#owner: https://api.github.com/users/Ashu-max

import json
from pprint import pprint
from statistics import mean
 
# read python dict back from the file
with open('/path-to-json/test-parse.json', 'rb') as json_file:
   data_response = json.load(json_file)
 
 
# use pretty print to format the data in JSON format
pprint(data_response)
 
#  calculate mean confidence for all
m = mean(
   a["confidence"] for r in data_response["results"] for a in r["alternatives"]
   )
print("Mean confidence:", m)