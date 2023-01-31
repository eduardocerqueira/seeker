#date: 2023-01-31T17:05:54Z
#url: https://api.github.com/gists/8c5db23b6059aca4eefedb8a44dd4e18
#owner: https://api.github.com/users/Marte77

import requests
from nslookup import Nslookup
import json
#WARNING
#THIS WILL REPLACE ALL YOUR EXISTING DNS RECORDS IN GO DADDY

DOMAIN_LOOKUP = "martinho.dynip.sapo.pt" # the ip correspondent to this domain will be used as the ip of the DOMAIN variable
DOMAIN = "martetm.eu"
meu_ip = Nslookup().dns_lookup(DOMAIN).response_full[0].split(' ')[-1]

base_url = 'https://api.godaddy.com'

api_key = 'YOUR_API_KEY'
api_secret = "**********"

headers = {'Authorization': "**********":'+api_secret}

novosupdates = []

novosupdates.append({
    'data':meu_ip,
    'name':'@',
    'ttl':3600,
})

headers['Content-Type'] = 'application/json'
r = requests.put(base_url + '/v1/domains/'+DOMAIN+'/records/A', headers=headers, data=json.dumps(novosupdates))
print(r.status_code)
print(r.text))
print(r.text)