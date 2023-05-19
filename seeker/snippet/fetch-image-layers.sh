#date: 2023-05-19T16:50:17Z
#url: https://api.github.com/gists/280a3771ebebfec01466493c823e37fa
#owner: https://api.github.com/users/Div123

#!/usr/bin/env bash
set -o errexit #fail on first error
set -o pipefail

./save-all-objects.sh
test -d layers || mkdir layers
test -d layers/v1 || mkdir layers/v1
test -d layers/v2 || mkdir layers/v2
test -d manifests || mkdir manifests
test -d manifests/v1 || mkdir manifests/v1
test -d manifests/v2 || mkdir manifests/v2
test -d images || mkdir images

namespace="$1"
imagestream="$2"
image="$3"

#while IFS=$'\t' read -r namespace name image; do
  test -f "images/${image}.json" || (echo "fetching ${namespace}/${imagestream}@${image}"; oc -n "${namespace}" get "isimage/${imagestream}@${image}" -o json > "images/${image}.json")
  dockerImageManifestMediaType="$(jq -crM '.image.dockerImageManifestMediaType' "images/${image}.json")"
  if [ "$dockerImageManifestMediaType" == "application/vnd.docker.distribution.manifest.v1+json" ]; then
    test -f "manifests/v1/${image}.json" || ( echo "Fetching manifest ${namespace}/${imagestream}/manifests/${image}"; curl -kfsSL -H "Authorization: Bearer $(oc whoami -t)" "https://docker-registry.pathfinder.gov.bc.ca/v2/${namespace}/${imagestream}/manifests/${image}" -o "manifests/v1/${image}.json" || echo "Error downloading ${namespace}/${imagestream}/manifests/${image}")
  else
    #test -f "manifests/v2/${image}.json" || curl -kfsSL -H "Authorization: Bearer $(oc whoami -t)" "https://docker-registry.pathfinder.gov.bc.ca/v2/${namespace}/${imagestream}/manifests/${image}" -o "manifests/v2/${image}.json"
    test -f "images/${image}-layers.txt" || jq -Mr ".image.dockerImageLayers // [] | .[].name" "images/${image}.json" > "images/${image}-layers.txt"
  fi
#done