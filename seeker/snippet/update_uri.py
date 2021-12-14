#date: 2021-12-14T16:58:04Z
#url: https://api.github.com/gists/13e180dd92c41cbab797949e6ab657af
#owner: https://api.github.com/users/NFTeez-Nutz

#!/usr/bin/env python3

import os
import json

base_dir = "/home/user/metadata"
img_hash = ""
for file in os.listdir(base_dir):
    if "json" not in file:
        continue
    id = file.split('.')[0]
    path = f"{base_dir}/{file}"
    # Remove .json from file name
    # os.rename(f"{base_dir}/{file}",f"{base_dir}/{id}")
    # path = f"{base_dir}/{id}"
    with open(path,'r+') as f:
        data = json.loads(f.read())
        f.seek(0)
        data['image'] = f"ipfs://{img_hash}/{id}.png"
        f.write(json.dumps(data))
        f.truncate()