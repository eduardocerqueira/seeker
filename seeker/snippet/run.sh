#date: 2024-04-08T17:03:00Z
#url: https://api.github.com/gists/6204bdae6d6e9b1d5ba3d0ec05a87503
#owner: https://api.github.com/users/teebow1e

#!/bin/sh

sudo docker pull ubuntu && sudo docker run -it $(sudo docker images ubuntu -q) /bin/sh