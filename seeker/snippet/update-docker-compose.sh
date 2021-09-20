#date: 2021-09-20T16:43:37Z
#url: https://api.github.com/gists/26d5eefafc9c5c524cc732d252f8ec78
#owner: https://api.github.com/users/timabell

#!/bin/sh

# tag grabber from: https://gist.github.com/lukechilds/a83e1d7127b78fef38c2914c4ececc3c
# original install instructions https://docs.docker.com/compose/install/

latest_tag=`curl --silent "https://api.github.com/repos/docker/compose/releases/latest"|grep '"tag_name":'|sed -E 's/.*"([^"]+)".*/\1/'`
target=~/Documents/programs/bin/docker-compose
curl -L "https://github.com/docker/compose/releases/download/$latest_tag/docker-compose-$(uname -s)-$(uname -m)" -o $target
chmod +x $target
