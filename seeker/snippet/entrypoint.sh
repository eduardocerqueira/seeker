#date: 2021-11-16T17:09:33Z
#url: https://api.github.com/gists/4c019d2b572453f0b9d18847874abd29
#owner: https://api.github.com/users/denibertovic

#!/bin/bash

# Add docker group
groupadd docker --gid ${DOCKER_GROUP_ID:-129}

# Add local user
# Either use the LOCAL_USER_ID if passed in at runtime or
# fallback
# The user is added to the docker group to be able to start containers

USER_ID=${LOCAL_USER_ID:-9001}

echo "Starting with UID : $USER_ID"
useradd --shell /bin/bash -G ${DOCKER_GROUP_ID:-129} -u $USER_ID -o -c "" -m user
export HOME=/home/user

exec /usr/local/bin/gosu user "$@"