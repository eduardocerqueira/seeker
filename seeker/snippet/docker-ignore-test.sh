#date: 2023-10-04T16:58:30Z
#url: https://api.github.com/gists/96aea872b0af1ee367b9f24055421eb0
#owner: https://api.github.com/users/AlexAtkinson

#!/usr/bin/env bash

dir=$(mktemp -d)
cd $dir > /dev/null

export DOCKER_BUILDKIT=1

echo -e "FROM node:20-alpine\nCOPY . tmp/" > Dockerfile_foo
cp Dockerfile_foo Dockerfile_bar
touch foo bar
echo "foo" > .dockerignore
echo "bar" > Dockerfile_bar.dockerignore

docker build -t foo -f Dockerfile_foo .
docker build -t bar -f Dockerfile_bar .

echo -- foo --
docker run foo ls tmp | grep '^foo\|^bar'
echo -- bar --
docker run bar ls tmp | grep '^foo\|^bar'

echo -e "\nTEST:   Observe how docker build handles .dockerignore files with BUILDKIT."
echo -e "\nRESULT: Docker build uses the <Dockerfile>.dockerignore and ignores .dockerignore."
echo -e "\nWARN:   Disabling BUILDKIT results in the <Dockerfile>.dockerignore file being ignored."

cd - > /dev/null
rm -rf $dir