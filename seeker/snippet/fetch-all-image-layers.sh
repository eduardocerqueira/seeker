#date: 2023-05-19T16:50:17Z
#url: https://api.github.com/gists/280a3771ebebfec01466493c823e37fa
#owner: https://api.github.com/users/Div123

#!/usr/bin/env bash
set -o errexit #fail on first error
set -o pipefail


./save-all-objects.sh
test -d layers || mkdir layers
test -d images || mkdir images

find images -empty -delete

jq -Mrc '.items[] | . as $item | .status.tags // [] | .[] | . as $tag | .items // [] | .[] | [$item.metadata.namespace, $item.metadata.name, $tag.tag, .image] | @tsv' all-image-streams.json |
while IFS=$'\t' read -r namespace name tag image; do
  ./fetch-image-layers.sh "${namespace}" "${name}" "${image}"
done