#!/bin/bash

echo "set github push"
git remote set-url --push origin https://$GITHUB_USERNAME:$GITHUB_TOKEN@github.com/eduardocerqueira/seeker
seeker $SEEKER_RUN
