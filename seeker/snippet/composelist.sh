#date: 2022-10-11T17:19:28Z
#url: https://api.github.com/gists/7f5a4eada042ebcbd2d707b1534baff7
#owner: https://api.github.com/users/ASKMAGICSHELL

#!/bin/bash

docker ps --filter "label=com.docker.compose.project" -q | xargs docker inspect --format='{{index .Config.Labels "com.docker.compose.project"}}'| sort | uniq
