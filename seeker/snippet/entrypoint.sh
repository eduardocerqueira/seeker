#date: 2023-05-11T17:01:23Z
#url: https://api.github.com/gists/2d1f4af8588a1c2a38ca25c3adec7148
#owner: https://api.github.com/users/thedomeffm

#!/bin/sh

env >> /etc/environment

# execute CMD
echo "$@"
exec "$@"
