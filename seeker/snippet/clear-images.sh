#date: 2022-12-27T16:51:57Z
#url: https://api.github.com/gists/c1dd7116f4bd8035b7424c467b319821
#owner: https://api.github.com/users/moimudi

#!/bin/bash
###########################################
#
# Simple Shell script to clean/remove all container/images
#
# The script will 
#  - first stop all running containers (if any),
#  - remove containers
#  - remove images
#  - remove volumes
#

# stop all running containers
echo '####################################################'
echo 'Stopping running containers (if available)...'
echo '####################################################'
docker stop $(docker ps -aq)

# remove all stopped containers
echo '####################################################'
echo 'Removing containers ..'
echo '####################################################'
docker rm $(docker ps -aq)


# remove all images
echo '####################################################'
echo 'Removing images ...'
echo '####################################################'
docker rmi $(docker images -q)

# remove all stray volumes if any
echo '####################################################'
echo 'Revoming docker container volumes (if any)'
echo '####################################################'
docker volume rm $(docker volume ls -q)

