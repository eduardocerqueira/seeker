#!/bin/bash

docker build -t seeker:latest -f Dockerfile . --network host

#cat 'doesnotexist.txt' 2>/dev/null || echo "NAO"
#
#docker images -q seeker:latest
#if [ $? -eq 0 ]; then
#  echo "OKOKOK"
#fi
#
##if [[ "$(docker images -q seeker:latest 2> /dev/null)" == "" ]]; then
##  docker rmi seeker --force
##fi
##
##docker build -t seeker -f Dockerfile . --network host
#echo
#
#echo "--- images built ---"
#docker image ls | grep -e 'seeker'
