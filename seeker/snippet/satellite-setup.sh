#date: 2023-07-03T17:04:21Z
#url: https://api.github.com/gists/0f746e42c132009eb8ed1eb6b705b5fd
#owner: https://api.github.com/users/elroncio

#!/bin/bash
# hammer defaults add --param-name organization_id --param-value 1

hammer content-view create --repository-ids 9,13 --organization-id 1 --name RHEL8-repo
hammer lifecycle-environment create --name Dev --prior-id 1 --organization-id 1
hammer lifecycle-environment create --name PreProd --prior-id 2 --organization-id 1
hammer lifecycle-environment create --name Prod --prior-id 3 --organization-id 1
hammer content-view publish --id 2  --organization-id 1
hammer content-view publish --id 2  --organization-id 1 --lifecycle-environment-ids 2,3
hammer activation-key create --auto-attach true --content-view RHEL8-repo --unlimited-hosts --lifecycle-environment-id 2 --name rhel8-key --organization-id 1