#date: 2023-05-11T16:57:48Z
#url: https://api.github.com/gists/f35d551730941e24bb6b803e7d751906
#owner: https://api.github.com/users/ericreeves

#!/bin/bash
#
# Quick and dirty script that wraps around the TFX CLI tool to list all Workspaces within an Orgainzation, and then list all remote state
# sharing for each Workspace.
#
# Acquire tfx here: https://tfx.rocks/
# TFX can be configured using environment variables TFE_HOSTNAME, TFE_ORGANIZATION, TFE_TOKEN
#

export TFE_HOSTNAME="app.terraform.io"
export TFE_ORGANIZATION="ericreeves-demo"
export TFE_TOKEN= "**********"

printf "+ %-50s + %s\n" $(printf -- '-%.0s' {1..50}) $(printf -- '-%.0s' {1..50})
printf "| %-50s | %s\n" "Source Workspace" "Sharing State With..."
printf "+ %-50s + %s\n" $(printf -- '-%.0s' {1..50}) $(printf -- '-%.0s' {1..50})
for WORKSPACE in $(tfx workspace list --json | jq -r '.[].Name'); do
    REMOTE_STATE_SHARING=$(tfx workspace show --name $WORKSPACE --json | jq -r '.["Remote State Sharing"][]' | paste -sd ",")
    printf "| %-50s | %s\n" $WORKSPACE $REMOTE_STATE_SHARING

done
printf "+ %-50s + %s\n" $(printf -- '-%.0s' {1..50}) $(printf -- '-%.0s' {1..50})