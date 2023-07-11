#date: 2023-07-11T17:10:11Z
#url: https://api.github.com/gists/b08e0d3e7ca7711db762d77fa4b1fddb
#owner: https://api.github.com/users/debdutdeb

#!/bin/bash

set -x

_tag=${1:-latest}

_dir="$(mktemp -d)"

trap "rm -rf $_dir" EXIT

pushd "$_dir" >/dev/null

cat <<EDOCKERFILE >Dockerfile
from rocketchat/rocket.chat:$_tag as upstream
from scratch
copy --from=upstream /app/bundle /rocket.chat
EDOCKERFILE

echo "info: preparing archive"
if ! _id="$(docker build -q .)"; then
	echo "error: failed to build new image"
	exit 1
fi

echo "info: image id $_id"
echo "info: saving image"

docker save "$_id" >rocketchat.tar
if ! [[ -f rocketchat.tar ]]; then
	echo "error: failed to save image"
	exit 1
fi

if ! tar xf rocketchat.tar; then
	echo "error: failed to extract archive"
	exit 1
fi

if ! _to_extract="$(jq '.[0].Layers[0]' -r manifest.json)"; then
	echo "error: no layer found in image"
	exit 1
fi

if ! tar xf "$_to_extract"; then
	echo "error: failed to extract main layer"
	exit 1
fi

if ! [[ -d rocket.chat ]]; then
	echo "error: no rocket.chat bundle found"
	exit 1
fi

if ! tar -czf "rocket.chat.$_tag.tar.gz" rocket.chat; then
	echo "error: failed to archive final archive"
	exit 1
fi

popd
mv "${_dir%/}/rocket.chat.$_tag.tar.gz" .

echo "info: your rocket.chat version $_tag archive is ready to be used"
