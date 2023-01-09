#date: 2023-01-09T16:56:08Z
#url: https://api.github.com/gists/53e2ce779255d7d1e30a101e1340c7b0
#owner: https://api.github.com/users/remuslazar

#!/usr/bin/env python3

import sys
import re
import zipfile


if len(sys.argv) != 2:
    print("Please pass .knxproj file as argument")

def ga(num):
    num = int(num)
    return f"{(num & 0x7800) >> 11}/{(num & 0x700) >> 8}/{num & 0xff}"

z = zipfile.ZipFile(sys.argv[1])
xml = next(n for n in z.namelist() if "0.xml" in n)
data = z.read(xml).decode("utf-8")
res = re.findall(r"GroupAddress.*Address=\"(\d+)\".*Name=\"([^\"]+)\".*Description=\"([^\"]+)\"",data)
with open("export.txt", "w") as ex:
    for r in res:
        ex.write(f"{ga(r[0]):<15}\t{r[1]}\t{r[2]}\n") 
