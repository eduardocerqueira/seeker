#date: 2022-05-18T17:00:06Z
#url: https://api.github.com/gists/69027eba89528da3d7a5b2cb6c056488
#owner: https://api.github.com/users/givemefoxes

import json

with open("export.json", encoding="utf8") as file:
    mem_export = file.read()
    data = json.loads(mem_export)

for i in data:
#    print(i["id"])
    f = open("converted\\" + i["id"] + ".md", "x", encoding="utf8")
    f.write(i["markdown"])