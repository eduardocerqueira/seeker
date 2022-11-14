#date: 2022-11-14T17:23:21Z
#url: https://api.github.com/gists/a10c0a71cf35f5f405e2d4d058e15cd6
#owner: https://api.github.com/users/phpmaps

import pandas as pd
import argparse
import json
import urllib.parse
from urllib.parse import urlparse

arr = []

def har(harfile_path):
    harfile = open(harfile_path)
    harfile_json = json.loads(harfile.read())
    i = 0

    for entry in harfile_json['log']['entries']:
        i = i + 1
        url = entry['request']['url']
        urlparts = urlparse(entry['request']['url'])
        size_bytes = entry['response']['content']['size']
        size_kilobytes = float(size_bytes)/1024
        t = entry['time']
        arr.append({
            'num': 1,
            'url': url,
            'size_kilobytes': size_kilobytes,
            'time_ms': t
        })
        

    print("Done reading HAR, starting excel file output.")

    df = pd.DataFrame.from_dict(arr)
    df.to_excel('network_latency_output.xlsx')

har('stage.har')