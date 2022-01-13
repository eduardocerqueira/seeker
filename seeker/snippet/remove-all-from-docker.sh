#date: 2022-01-13T17:14:49Z
#url: https://api.github.com/gists/0aeca5208229468ac4670ee915656f1b
#owner: https://api.github.com/users/avostokov

# Stop all containers
docker stop `docker ps -qa`

# Remove all containers
docker rm `docker ps -qa`

# Remove all images
docker rmi -f `docker images -qa `

# Remove all volumes
docker volume rm $(docker volume ls -q)

# Remove all networks
docker network rm `docker network ls -q`

# Your installation should now be all fresh and clean.

# The following commands should not output any items:
# docker ps -a
# docker images -a 
# docker volume ls

# The following command show only show the default networks:
# docker network ls

