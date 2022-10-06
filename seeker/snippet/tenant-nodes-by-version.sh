#date: 2022-10-06T17:16:08Z
#url: https://api.github.com/gists/fbdfa5ec2076d08c55b0ce1a73357dae
#owner: https://api.github.com/users/gozer

#!/bin/bash

set -e -o pipefail

TENANT=$1
VERSION=$2

CACHE=$TENANT-versions.json

dy -t "tenant-${TENANT}.main" query teleport --sort-key "begins_with teleport/nodes/" -o raw \
  |  jq '. [] | .Value.B | @base64d | fromjson | { kind: .kind, version: .spec.version, hostname: .spec.hostname}' \
  | jq --slurp | tee "$CACHE"

if [[ -n "$VERSION" ]]; then
  jq ".[] | select(.version|test(\"$VERSION\"))" < "$CACHE"
fi
