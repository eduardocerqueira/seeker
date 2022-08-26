#date: 2022-08-26T16:45:34Z
#url: https://api.github.com/gists/720a2bfad3ad75d8650791df31e7f964
#owner: https://api.github.com/users/SimonDMC

# A simple python script using the YouTube API to add up all views on videos of people playing my maps
# Created by SimonDMC, published under the MIT license

import requests
import json

total_views = 0

api_key = 'YOUTUBE API KEY'
ids = ["OZvQUuGbyX8", "cYaUqIbzsJk", "34n68sex6PQ", "68UTeIfTHw8", "Vx4RdktJuTc", "EaVQ3QuSo3g", "atRPoO6kbr0", "flTsbzJ3otE", "3NxDe-XxolA", "W4qFqqLOCQ8", "D40emi8DixI", "yMD8WvRjP9U", "3G9hlg_xDro", "Hr9D0pDq9DU", "irgWcp27FV0", "5SvP4CI76Ic", "bucVaONi3UU", "m4qomF_2fdo", "Du1vGA_z9ZE", "vYp3-tlxSUU", "qi-mxwQFvZk", "snnvTPnFs0I", "q7uGAhi-INo", "kqR9T6E1jBM", "HFs77QIXmdk", "YniM1mcM5f0", "4EpeM43fBfY", "ZwfBoh4K2_s", "UIlbd2f95BE", "NJZKNrL19dc", "XW60QgFMDqM", "7eW2_6O656k", "fcawnkNDOYQ", "l6XWej2l8xo", "4yeTSKw_zDM", "IN9RkweKMeo", "0ww3B29nZJg", "xEIA08VdGbc", "oKYiMRd6T0M", "p-CwZu0Da_o", "tieBsH0_ZAU", "BCM2Pb3kMB8", "SmgF4CJDCKM", "477nC4ArhE4", "pZ7HvCWVWyU", "e8bwfajnipA", "JQ7CPqISLe4", "xnrpQjo_SSo"]

for yt_id in ids:
    # compile GET url and send request
    request = 'https://www.googleapis.com/youtube/v3/videos?id=' + yt_id + '&key=' + api_key + '&part=statistics'
    res = requests.get(request)
    response = json.loads(res.text)
    # extract views from response and add to total
    total_views += int(response['items'][0]['statistics']['viewCount'])

print(total_views)
