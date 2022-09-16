#date: 2022-09-16T21:25:21Z
#url: https://api.github.com/gists/f32b5963ecf0492f863151f629aa06d0
#owner: https://api.github.com/users/sidneyflima

#!/bin/bash

# Script used to run bitnami/redis:6.0.16 image using docker run commands for testing in development environment.
# It will create a new container named 'redis_server'.

# Replace <-- insert_redis_password_here --> by redis auth password or ignore

docker stop redis_server
docker rm redis_server
docker run -d \
	-e REDIS_PASSWORD= "**********"
	-p 6379:6379 \
	--name redis_server \
	bitnami/redis:6.0.16

	bitnami/redis:6.0.16
