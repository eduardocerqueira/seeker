#date: 2023-05-19T16:50:17Z
#url: https://api.github.com/gists/280a3771ebebfec01466493c823e37fa
#owner: https://api.github.com/users/Div123

#!/usr/bin/env bash
set -o errexit #fail on first error
set -o pipefail

./cache-stdout oc get imagestreams -o json | jq -r '.items[] | {imageStream:.metadata.name, image:.status.tags[]?.items[]?.image} | (.imageStream + "@" + .image)' | xargs -I {} ./cache-stdout oc get 'imagestreamimage/{}' -o json |  jq -r '.image.dockerImageLayers[] | {name: .name, size: .size}' | jq -sr 'unique_by(.name) | .[].size' | awk '{ sum += $1 } END { print (sum / 1024 / 1024 / 1024) "GB" }'
