#date: 2025-03-17T17:01:32Z
#url: https://api.github.com/gists/ae610cd4f0fdc2912da4ef4a21e626bc
#owner: https://api.github.com/users/RahulNavneeth

#!/bin/bash

for i in "$@"; do
	git update-index --assume-unchanged ${i}
done
