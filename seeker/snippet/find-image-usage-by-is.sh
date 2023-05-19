#date: 2023-05-19T16:50:17Z
#url: https://api.github.com/gists/280a3771ebebfec01466493c823e37fa
#owner: https://api.github.com/users/Div123

#!/usr/bin/env bash
# Example:
# ./find-image-usage-by-is.sh openshift/jenkins-2-centos7

./save-all-objects.sh
test -d layers || mkdir layers
test -d images || mkdir images

_ns_is="$1"
_ns="$(cut -d / -f1 <<< "$_ns_is")"
_is="$(cut -d / -f2 <<< "$_ns_is")"

jq -Mrc --arg ns "${_ns}" --arg is "${_is}" '.items[] | select(.metadata.namespace == $ns and .metadata.name == $is) | . as $item | .status.tags // [] | .[] | . as $tag | .items[] | [$item.metadata.namespace, $item.metadata.name, $tag.tag, .image] | @tsv' all-image-streams.json |
while IFS=$'\t' read -r namespace name tag image; do
  toplayer="$(head -1 "images/${image}-layers.txt")"
  echo "top layer of ${image} is ${toplayer}"
  find images -type f -name '*-layers.txt' | grep -v "images/${image}-layers.txt" |  xargs grep -oh -F "${toplayer}" | wc -l
done
