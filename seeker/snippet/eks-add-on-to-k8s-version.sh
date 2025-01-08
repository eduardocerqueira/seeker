#date: 2025-01-08T17:08:59Z
#url: https://api.github.com/gists/612e512976a3bcde7185a115a5f73bbf
#owner: https://api.github.com/users/dims

#!/bin/bash

aws eks describe-addon-versions | jq -r '.addons[] | 
 {
  addon: .addonName,
  versions: [
    .addonVersions[] | {
      version: .addonVersion,
      kubernetes_versions: [
        .compatibilities[].clusterVersion
      ] | sort | unique
    }
  ]
}'