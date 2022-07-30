#!/bin/bash

[ -z ${GITHUB_TOKEN} ] && echo "GITHUB_TOKEN is mandatory, not defined"
docker run -e GITHUB_TOKEN="$GITHUB_TOKEN" -e GITHUB_USERNAME="eduardocerqueira" -e GITHUB_EMAIL="eduardomcerqueira@gmail.com" -it seeker /bin/bash
