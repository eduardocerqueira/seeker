#date: 2023-07-31T16:59:10Z
#url: https://api.github.com/gists/4264722e90bb8ffb69d1ceb7f489afb7
#owner: https://api.github.com/users/milktea736

#!/bin/bash

function delete_targets() {
    # target="team-X-mails"
    # yq   'del(.route.routes[] | select(.receiver==env(target)) )' alert.yml
    # yq  'del(.route.routes[] | select(.receiver | test(env(target)) ) )' alert.yml
    yq  'del(.route.routes[] | select(.receiver=="*mails" ) )' alert.yml > tmp.yml
}

function grep_url(){
    # extrac the host from the url, the host name contains "your-webhook"
    grep -oE 'http://your-webhook[^ ]+example\.com'  alert.yml
}

function merge_yml() {
    yq '. *d load("patch.yml")' tmp.yml
}

function template_yml() {
    # use environment variables to template the yml file
    yq '(.. | select(tag == "!!str")) |= envsubst ' patch.yml
}