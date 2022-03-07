#date: 2022-03-07T17:04:33Z
#url: https://api.github.com/gists/f19d2dd2e1389e2a6da4a304bfec656a
#owner: https://api.github.com/users/frafra

#!/bin/bash
#
# Build a container with the required command and run it

set -eu

docker_like() {
    if command -v podman &> /dev/null
    then
       podman "$@"
    elif command -v rootlesskit &> /dev/null
    then
      rootlesskit docker "$@"
    elif command -v docker &> /dev/null
    then
      docker "$@"
    else
      echo "Cannot run $@"
    fi
}

run_in_container() {
	exe=$1
	docker_like build --tag $exe --build-arg exe=$exe - <<- 'EOF'
	FROM fedora:36
	ARG exe
	RUN dnf install -y $(dnf provides -q {/usr/bin,/usr/sbin}/$exe | awk '{ if (NR%5==1) pkg=$1 } END { print pkg }')
	EOF
	docker_like run --rm -it -v $HOME:/root --workdir /root $exe $*
}

run_in_container "$@"