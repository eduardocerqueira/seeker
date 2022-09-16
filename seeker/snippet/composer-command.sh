#date: 2022-09-16T23:05:23Z
#url: https://api.github.com/gists/ba052cd8f380c38f98e4ee396049c537
#owner: https://api.github.com/users/MatthewDaniels

#!/bin/bash

# use this to run composer commands in a container (rather than on the host machine)
# to execute: 
#    ./composer-command.sh {{ composer commands }}
# the {{ composer commands }} above get passed to composer for execution (eg: "install")
# 
# ./composer-command.sh install --dev
#
# Install Laravel (run this from a parent directory)
# ./composer-command.sh create-project laravel/laravel example-app --ignore-platform-reqs


# Some suggested composer options
#   '--dev' when using this for dev work 
#   '--prefer-dist --optimize-autoloader' for production
#   '--ignore-platform-reqs' for all platforms so composer does not check the "platform" (as that is the composer docker container)


# Info / more options / ideas
# @see: https://hub.docker.com/_/composer

docker run --rm --interactive --tty \
  --volume $PWD:/app \
  composer "$@" --verbose