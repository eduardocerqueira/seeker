#date: 2022-07-31T02:47:37Z
#url: https://api.github.com/gists/53facc48d3d02e2f2dfa44e1b0f5dd4c
#owner: https://api.github.com/users/p7e4

import requests

s = requests.Session()

user = "p7e4"

data = []
for page in range(10):
	p = s.get(f"https://api.github.com/users/{user}/starred?per_page=100&page={page}").json()
	if not p: break
	for i in p:
		data.append(i["html_url"])

with open("stars.txt", "w") as f:
	f.write("\n".join(data) + "\n")
