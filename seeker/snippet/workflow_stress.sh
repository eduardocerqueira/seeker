#date: 2023-12-04T16:54:42Z
#url: https://api.github.com/gists/dc01a5d9961cfac1ffacbdcef669b230
#owner: https://api.github.com/users/bduffany

#!/bin/bash
set -euo pipefail

: "${API_KEY:=}"
: "${TARGET:=grpc://localhost:1985}"
: "${N:=10}"

TOPLEVEL_IID=be24c0b8-8742-4cda-b58e-25117ac9dcf8
echo "Invocation ID for cancellation: $TOPLEVEL_IID"

REMOTE_CONFIG=""
if [[ "$TARGET" =~ \.dev$ ]]; then
  REMOTE_CONFIG="--config=remote-dev"
fi
if [[ "$TARGET" =~ \.io$ ]]; then
  REMOTE_CONFIG="--config=remote"
fi

trap 'kill -KILL $(jobs -p) &>/dev/null' EXIT

for _ in $(seq 1 "$N"); do
  API_KEY_ARG=()
  if [[ "$API_KEY" ]]; then
    API_KEY_ARG+=(--remote_header=x-buildbuddy-api-key="$API_KEY")
  fi
  # IID=$(uuidgen)
  # echo "Invocation will be available at https://app.buildbuddy.dev/invocation/$IID?queued=true"
  bb --verbose=1 execute \
    --invocation_id="$TOPLEVEL_IID" \
    "${API_KEY_ARG[@]}" \
    --exec_properties=Pool=workflows \
    --exec_properties=workload-isolation-type=firecracker \
    --exec_properties=container-image=docker://gcr.io/flame-public/rbe-ubuntu20-04-workflows@sha256:271e5e3704d861159c75b8dd6713dbe5a12272ec8ee73d17f89ed7be8026553f \
    --exec_properties=recycle-runner=true \
    --exec_properties=workflow-id=WF0000000000 \
    --exec_properties=EstimatedMemory=10GB \
    --exec_properties=EstimatedFreeDisk=20GB \
    --exec_properties=EstimatedCPU=3000m \
    --exec_properties=init-dockerd=true \
    --action_env=GIT_BRANCH=master \
    --action_env=GIT_BASE_BRANCH=master \
    --action_env=GIT_REPO_DEFAULT_BRANCH=master \
    --remote_executor="$TARGET" \
    -- \
    bash -ec '
export HOME=/root
export PATH="/bin:/usr/bin:/usr/sbin:/sbin:/usr/local/bin"

cd /root
mkdir -p workspace
cd workspace

if ! [[ -e buildbuddy ]]; then
  git clone --filter=blob:none https://github.com/buildbuddy-io/buildbuddy
fi
cd buildbuddy

git pull
bazelisk build //... '"$REMOTE_CONFIG"' '"${API_KEY_ARG[*]}"'

' &
done

echo "Waiting..."
wait
