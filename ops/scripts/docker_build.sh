#!/bin/bash

docker images -q seeker:latest
if [ $? -eq 0 ]; then
  docker rmi seeker --force
fi

docker build -t seeker -f Dockerfile . --network host
echo

echo "--- images built ---"
docker image ls | grep -e 'seeker'
