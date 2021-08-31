#!/bin/bash

if [[ "$(docker images -q seeker:latest 2> /dev/null)" == "" ]]; then
  docker rmi seeker --force
fi

docker build -t seeker -f Dockerfile . --network host
echo

echo "--- images built ---"
docker image ls | grep -e 'seeker'
