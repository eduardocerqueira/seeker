#date: 2022-12-16T16:50:05Z
#url: https://api.github.com/gists/df126652cf39b93af8eea508e17a9f13
#owner: https://api.github.com/users/imflikk

import sys
import re

if len(sys.argv) < 2:
	print("Please provide an input file with saved CrackMapExec output.")
	exit(1)

CME_FILE = sys.argv[1]
pattern = r"\\x[0-9a-f]+[[0-9]+m"

with open(CME_FILE, "rb") as f:
	lines = f.readlines()

	for line in lines:
		line = str(line)[2:-1]
		line = re.sub(pattern, '', line)
		print(line)