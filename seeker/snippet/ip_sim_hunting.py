#date: 2023-01-11T16:57:44Z
#url: https://api.github.com/gists/8ab0d7a6f2d586d56494634389a89c56
#owner: https://api.github.com/users/superducktoes

import requests
import sys

api_key = ""
limit = 10 # can change for more

if(len(sys.argv) < 2):
    print("need an IP")
    quit()
headers = {
    'accept': 'application/json',
    'key': api_key,
}

similar_ips = []
similar_ips_string = ""
web_paths = []
web_paths_string = ""

r = requests.get("https://api.greynoise.io/v3/similarity/ips/{}?limit={}".format(sys.argv[1], limit), headers=headers).json()

if("similar_ips" not in r):
    print("no similar ips found")
    sys.exit()
    
for i in r["similar_ips"]:
    similar_ips.append(i["ip"])
    gnql = requests.get("https://api.greynoise.io/v2/noise/context/{}".format(i["ip"]), headers=headers).json()
    if("raw_data" in gnql and "paths" in gnql["raw_data"]["web"]):
        for j in gnql["raw_data"]["web"]["paths"]:
            if j not in web_paths and j != "/":
                web_paths.append(j)
    
print("Similar IPs: ")
print(*similar_ips, sep=", ")

print("\nWeb Paths: ")
print(*web_paths, sep=", ")

base_search = "index=main sourcetype=access_combined"
for i in similar_ips:
    similar_ips_string += (i + ", ")
similar_ips_string = similar_ips_string[:-2]

print("\nSplunk Search - IPs: ")
#print(base_search[:-3])
print(base_search + " clientip IN ({})".format(similar_ips_string))
      
base_search_paths = "index=main sourcetype=access_combined"

for i in web_paths:
    web_paths_string += (i + ", ")
web_paths_string = web_paths_string[:-2]

print("\nSplunk Search - Paths: ")
print(base_search_paths + " uri IN ({})".format(web_paths_string))