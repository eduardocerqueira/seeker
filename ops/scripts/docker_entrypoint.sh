#!/bin/bash

echo "Entrypoint"
git remote set-url --push origin https://$GITHUB_USERNAME:$GITHUB_TOKEN@github.com/eduardocerqueira/seeker
git config -l
