#date: 2022-09-16T21:28:34Z
#url: https://api.github.com/gists/21c12f7043c8f3554f399c02fe885e22
#owner: https://api.github.com/users/sidneyflima

#!/bin/bash

# Script used to create a new RabbitMQ container using docker run command for testing in development environment.
# It will create a new container named 'rabbitmq-server' and it will create a new volume called 'rabbitmq-server-volume' for RabbitMQ data folder (/var/lib/rabbitmq).

docker stop rabbitmq-server
docker rm rabbitmq-server

docker volume create rabbitmq-server-volume

docker run -d \
	--hostname rabbitmq-server-host \
	--name rabbitmq-server \
	-e RABBITMQ_DEFAULT_USER=guest \
	-e RABBITMQ_DEFAULT_PASS=guest \
	-p 5672:5672 \
	-p 15672:15672 \
	-v rabbitmq-server-volume:/var/lib/rabbitmq \
	rabbitmq:3.10.7-management \
