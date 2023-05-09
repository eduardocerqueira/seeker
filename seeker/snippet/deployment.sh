#date: 2023-05-09T17:04:01Z
#url: https://api.github.com/gists/bd16da2bff06295bafb28707c3db7cd7
#owner: https://api.github.com/users/raksit31667

#! /usr/bin/env bash

set -eou pipefail

az deployment sub create \
  --location <your-azure-location> \
  --name <your-deployment-name> \
  --template-file <path/to/template.bicep> \
  --parameters <path/to/parameter.json>