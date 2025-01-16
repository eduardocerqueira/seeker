#date: 2025-01-16T17:05:34Z
#url: https://api.github.com/gists/fae3e2aec866732f83ef65013186411e
#owner: https://api.github.com/users/guidorice

#!/usr/bin/bash

# docker image names and version tags are here: https://github.com/postgis/docker-postgis
# remember: "**********"

mkdir -p ~/pgdata
 
docker run --detach \
       --name postgres-dev \
       -p 5432:5432 \
       -e POSTGRES_PASSWORD= "**********"
       -e PGDATA=/var/lib/postgresql/data/pgdata \
       -v /home/user/pgdata:/var/lib/postgresql/data \
       postgis/postgis:17-3.5
stgis:17-3.5
