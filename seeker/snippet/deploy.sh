#date: 2021-11-11T17:10:44Z
#url: https://api.github.com/gists/57e1b00328b06a7eb69593ea6652e7a4
#owner: https://api.github.com/users/miminar

#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

SNO_NM_CONNECTION_FILE=./nm-system-connections/eno1.nmconnection
#WORKER_NM_CONNECTION_FILE=./nm-system-connections/sap-eno24.nmconnection
ISO=./rhcos-4.9.0-x86_64-live.x86_64.iso
# API will be available at https://api.$CLUSTER_NAME.$DOMAIN:6443
CLUSTER_NAME=snoplusone
DOMAIN=ocp.vslen
# assuming http server runs on the same host this script is run
HTTP_SERVE_ROOT=/srv/html/rhcos
IMAGE_SERVE="$HTTP_SERVE_ROOT/images"
IGNITION_SERVE="$HTTP_SERVE_ROOT/$CLUSTER_NAME/ignition"

function patchIgnitionFile() {
    local nmconfile="$1"
    local ignFn="$2"
    local content
    content="$(base64 -w0 <"$nmconfile")"
    jq --arg ifcfgContent "$content" --arg ifcfgFileName "$(basename "$nmconfile")" \
        '.storage |= (.files |= ((. // []) + [{
                "contents": {
                  "source": ("data:text/plain;charset=utf-8;base64," + $ifcfgContent),
                  "verification": {}
                },
                "filesystem": "root",
                "mode": 384,
                "path": ("/etc/NetworkManager/system-connections/" + $ifcfgFileName)
            }]))' "$ignFn" | sponge "$ignFn"
}

function getRootCA() {
    local bootstrapIgn="$1"
    jq -r '.storage.files[] | select(.path == "/opt/openshift/tls/root-ca.crt") | .contents.source' \
        "$bootstrapIgn" | sed 's#^data:text/plain;charset=utf-8;base64,##' | base64 -d
}

function genWorkerIgn() {
    local bootstrapIgn="$1"
    local rootCA
    rootCA="$(getRootCA "$bootstrapIgn")"
    jq -n --arg rootCA "$rootCA" --arg clusterName "$CLUSTER_NAME" --arg domain "$DOMAIN" '{
          "ignition": {
            "config": {
              "merge": [
                {
                  "source": ("https://api-int." + $clusterName +"."+ $domain + ":22623/config/worker")
                }
              ]
            },
            "security": {
              "tls": {
                "certificateAuthorities": [
                  {
                    "source": ("data:text/plain;charset=utf-8;base64," + ($rootCA | @base64))
                  }
                ]
              }
            },
            "version": "3.1.0"
          }
        }'
}


if [[ -e "$CLUSTER_NAME" ]]; then
    rm -rf ./"$CLUSTER_NAME"
    echo -n ''
fi
mkdir "$CLUSTER_NAME"
ln -vf install-config.yaml "$CLUSTER_NAME"/
cp -a install-config.yaml "$CLUSTER_NAME"/install-config.bak.yaml
openshift-install create manifests --dir=./"$CLUSTER_NAME"
openshift-install create single-node-ignition-config --dir=./"$CLUSTER_NAME"
patchIgnitionFile "$SNO_NM_CONNECTION_FILE" ./"$CLUSTER_NAME"/bootstrap-in-place-for-live-iso.ign
cp -v ./"$CLUSTER_NAME"/bootstrap-in-place-for-live-iso.ign "$IGNITION_SERVE"/
cp -v "$ISO" "$CLUSTER_NAME"-4.9-ctrl.iso
coreos-installer iso ignition embed -fi ./"$CLUSTER_NAME"/bootstrap-in-place-for-live-iso.ign \
    "$CLUSTER_NAME"-4.9-ctrl.iso
mv -v "$CLUSTER_NAME"-4.9-ctrl.iso "$IMAGE_SERVE/"

# generate worker files
genWorkerIgn ./"$CLUSTER_NAME"/bootstrap-in-place-for-live-iso.ign >./"$CLUSTER_NAME"/worker.ign
# For an unknown reason, the network configuration file did not work in our case.
    #patchIgnitionFile "$WORKER_NM_CONNECTION_FILE" ./"$CLUSTER_NAME"/worker.ign
    #cp -v "$ISO" "$CLUSTER_NAME"-4.9-worker.iso
    #coreos-installer iso ignition embed -fi ./"$CLUSTER_NAME"/worker.ign \
    #    "$CLUSTER_NAME"-4.9-worker.iso
    #mv -v "$CLUSTER_NAME"-4.9-worker.iso "$IGNITION_SERVE"/
cp -v "$ISO" "$IMAGE_SERVE/$CLUSTER_NAME"-4.9-worker.iso
cp -v ./"$CLUSTER_NAME"/worker.ign "$IGNITION_SERVE"/
