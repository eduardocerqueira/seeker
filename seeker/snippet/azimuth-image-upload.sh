#date: 2024-07-02T17:05:37Z
#url: https://api.github.com/gists/7a732d45ca98758eedb3fc4014619433
#owner: https://api.github.com/users/mkjpryor

#!/usr/bin/env bash

#####
# This script uploads the images for the specified azimuth-images version to the
# authenticated OpenStack project
#
# Before executing this script, OpenStack credentials for the target project suitable
# for use with the OpenStack CLI should be exported into the environment
#
# Requires curl, jq and the OpenStack CLI to be available in PATH
#
# NOTE: This script uploads the images using the same QCOW2 format that they are
#       distributed in
#       If required, they could be converted using qemu-img between download and upload 
#####

set -eo pipefail


echo "[INFO ] fetching latest azimuth-images version"
LATEST="$(curl -fsSL https://api.github.com/repos/stackhpc/azimuth-images/releases/latest | jq -r '.name')"
VN="${1:-$LATEST}"

echo "[INFO ] fetching manifest for $VN"
MANIFEST="$(curl -fsSL https://github.com/stackhpc/azimuth-images/releases/download/${VN}/manifest.json)"

for IMG in $(jq -r '. | keys | .[]' <<< "$MANIFEST"); do
  echo "[INFO ] processing image - $IMAGE_NAME"
  IMAGE_NAME="$(jq -r ".\"$IMG\".name" <<< "$MANIFEST")"
  IMAGE_FNAME="$IMAGE_NAME.qcow2"
  IMAGE_URL="$(jq -r ".\"$IMG\".url" <<< "$MANIFEST")"

  echo "[INFO ]   downloading image"
  curl -Lo "$IMAGE_FNAME" --progress-bar "$IMAGE_URL"

  echo "[INFO ]   uploading image to OpenStack"
  IMAGE_ID="$(
    openstack image create \
      --progress \
      --community \
      --container-format bare \
      --disk-format qcow2 \
      --file "$IMAGE_FNAME" \
      --format value \
      --column id \
      "$IMAGE_NAME"
  )"
  echo "[INFO ]   created image - $IMAGE_ID"

  rm "$IMAGE_FNAME"
done
