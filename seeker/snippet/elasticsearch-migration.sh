#date: 2025-06-24T17:08:17Z
#url: https://api.github.com/gists/805b3f9bc6e8366b8b9349416ab82108
#owner: https://api.github.com/users/sebagarayco

#!/bin/bash
# ==============================================================================
# Elasticsearch Index Migration Script (Snapshot & Restore)
#
# This script migrates selected non-hidden indices from a source Elasticsearch
# cluster to a target cluster using snapshot and restore functionality.
#
# Features:
# - Interactive index selection (including wildcard patterns or "all")
# - Excludes hidden indices (starting with .)
# - Sets source indices to read-only before snapshot
# - Validates snapshot repository existence in both clusters
# - Creates a named snapshot in the source cluster
# - Waits for the snapshot to appear in the target
# - Optionally deletes existing indices on the target before restore
# - Restores selected indices to the target cluster
# - Optionally re-enables write access on the source indices after restore
#
# Authentication is handled via environment variables or interactive prompts.
# ==============================================================================

set -euo pipefail

##############################################
# 🔧 Configurable Defaults
##############################################
SOURCE_ES_URL="https://elasticsearch.source.com"
TARGET_ES_URL="https://elasticsearch.target.com"
REPO_NAME="backup-repository-name"
MAX_RETRIES=12
RETRY_INTERVAL=10  # seconds

##############################################
# 🔐 Prompt for credentials (fallback to env)
##############################################
read -p "Elasticsearch username (default: elastic): " ES_USER
ES_USER=${ES_USER:-elastic}

# Get SOURCE password
if [ -z "${SOURCE_PASS:-}" ]; then
  read -s -p "Password for SOURCE cluster: "**********"
  echo
else
  echo "🔐 Using SOURCE_PASS from environment."
fi

# Get TARGET password
if [ -z "${TARGET_PASS:-}" ]; then
  read -s -p "Password for TARGET cluster: "**********"
  echo
else
  echo "🔐 Using TARGET_PASS from environment."
fi

SOURCE_AUTH_HEADER="Authorization: Basic $(echo -n "${ES_USER}:${SOURCE_PASS}" | openssl base64)"
TARGET_AUTH_HEADER="Authorization: Basic $(echo -n "${ES_USER}:${TARGET_PASS}" | openssl base64)"

##############################################
# 🏷️ Generate snapshot name with SOURCE host prefix
##############################################
SOURCE_HOSTNAME=$(echo "$SOURCE_ES_URL" | sed -E 's|https?://([^:/]+).*|\1|')
SNAPSHOT_PREFIX=$(echo "$SOURCE_HOSTNAME" | cut -d '.' -f 1)
SNAPSHOT_NAME="${SNAPSHOT_PREFIX}-snapshot_$(date +%Y%m%d_%H%M%S)"

##############################################
# 📦 Validate snapshot repository on both clusters
##############################################
validate_repo() {
  local cluster_url=$1
  local cluster_name=$2
  local auth_header=$3

  echo "🔎 Validating snapshot repository '$REPO_NAME' on $cluster_name cluster..."
  response=$(curl -s -o /dev/null -w "%{http_code}" -X GET "$cluster_url/_snapshot/$REPO_NAME" -H "$auth_header")

  if [ "$response" -ne 200 ]; then
    echo "❌ Snapshot repository '$REPO_NAME' not found or not accessible on $cluster_name cluster ($cluster_url)"
    exit 1
  else
    echo "✅ Repository '$REPO_NAME' found on $cluster_name."
  fi
}

validate_repo "$SOURCE_ES_URL" "SOURCE" "$SOURCE_AUTH_HEADER"
validate_repo "$TARGET_ES_URL" "TARGET" "$TARGET_AUTH_HEADER"

##############################################
# 🔎 Function: Check cluster health
##############################################
check_cluster_health() {
  local cluster_url=$1
  local cluster_name=$2
  local auth_header=$3

  echo "📊 Checking cluster health: $cluster_name"
  curl -s -X GET "$cluster_url/_cluster/health" \
    -H "$auth_header" -H 'Content-Type: application/json' | jq '.status'
}

##############################################
# 🟢 Start Migration
##############################################
echo "🔎 Checking pre-migration cluster health..."
check_cluster_health "$SOURCE_ES_URL" "SOURCE" "$SOURCE_AUTH_HEADER"
check_cluster_health "$TARGET_ES_URL" "TARGET" "$TARGET_AUTH_HEADER"

##############################################
# 📋 Select indices to migrate
##############################################
ALL_INDICES=$(curl -s -X GET "$SOURCE_ES_URL/_cat/indices?h=index" \
  -H "$SOURCE_AUTH_HEADER" | grep -v '^\.')  # exclude hidden

ALL_INDICES_SORTED=$(echo "$ALL_INDICES" | sort)

echo "📋 Available non-hidden indices in SOURCE cluster:"
echo "$ALL_INDICES_SORTED"

echo
read -p "Enter index patterns to migrate (e.g. clientA-*, clientB-* or type 'all'): " INDEX_PATTERNS_RAW

if [[ "$INDEX_PATTERNS_RAW" == "all" ]]; then
  MATCHING_INDICES="$ALL_INDICES_SORTED"
else
  IFS=',' read -ra INDEX_PATTERNS <<< "$INDEX_PATTERNS_RAW"

  MATCHING_INDICES=""
  for pattern in "${INDEX_PATTERNS[@]}"; do
    matched=$(echo "$ALL_INDICES" | grep -E "^${pattern//\*/.*}$" || true)
    MATCHING_INDICES+="$matched"$'\n'
  done
fi

MATCHING_INDICES=$(echo "$MATCHING_INDICES" | sort -u | grep -v '^$' | tr '\n' ',' | sed 's/,$//')

if [ -z "$MATCHING_INDICES" ]; then
  echo "❌ No indices matched the entered patterns. Exiting."
  exit 1
fi

echo "📄 Indices selected for migration:"
echo "$MATCHING_INDICES" | tr ',' '\n'

##############################################
# 🔐 Set indices to read-only
##############################################
echo "🔐 Setting selected indices to read-only..."
for index in $(echo "$MATCHING_INDICES" | tr ',' ' '); do
  curl -s -X PUT "$SOURCE_ES_URL/$index/_settings" \
    -H "$SOURCE_AUTH_HEADER" \
    -H 'Content-Type: application/json' \
    -d '{"settings":{"index.blocks.write":true}}' > /dev/null
done

##############################################
# 📸 Create snapshot
##############################################
echo "📸 Creating snapshot: $SNAPSHOT_NAME"
curl -s -X PUT "$SOURCE_ES_URL/_snapshot/$REPO_NAME/$SNAPSHOT_NAME?wait_for_completion=true" \
  -H "$SOURCE_AUTH_HEADER" \
  -H 'Content-Type: application/json' \
  -d "{\"indices\":\"$MATCHING_INDICES\",\"include_global_state\":false}"

##############################################
# ⏳ Wait until snapshot is visible on TARGET
##############################################
echo "⏳ Waiting for snapshot to appear in TARGET cluster..."
retry=0
while true; do
  SNAPSHOT_FOUND=$(curl -s -X GET "$TARGET_ES_URL/_snapshot/$REPO_NAME/$SNAPSHOT_NAME" \
    -H "$TARGET_AUTH_HEADER" | jq -r '.snapshots[]?.snapshot // empty')

  if [ "$SNAPSHOT_FOUND" == "$SNAPSHOT_NAME" ]; then
    echo "✅ Snapshot $SNAPSHOT_NAME is now available on target."
    break
  fi

  if (( retry >= MAX_RETRIES )); then
    echo "❌ Snapshot not found on target after $MAX_RETRIES retries. Exiting."
    exit 1
  fi

  echo "🔁 Not found yet. Retrying in $RETRY_INTERVAL seconds... ($((retry+1))/$MAX_RETRIES)"
  sleep $RETRY_INTERVAL
  ((retry++))
done

##############################################
# 🗑️ Optionally delete target indices before restore
##############################################
echo
read -p "⚠️ Do you want to delete matching indices in the TARGET cluster before restore? [y/N]: " DELETE_TARGET_INDICES
DELETE_TARGET_INDICES=$(echo "$DELETE_TARGET_INDICES" | tr '[:upper:]' '[:lower:]')

if [[ "$DELETE_TARGET_INDICES" =~ ^(y|yes)$ ]]; then
  echo "🗑️ Deleting existing indices in TARGET..."
  for index in $(echo "$MATCHING_INDICES" | tr ',' ' '); do
    curl -s -X DELETE "$TARGET_ES_URL/$index" \
      -H "$TARGET_AUTH_HEADER" \
      -H 'Content-Type: application/json' > /dev/null
  done
  echo "✅ Existing indices deleted on TARGET."
else
  echo "ℹ️ Existing indices on TARGET will remain untouched."
fi

##############################################
# 📦 Restore snapshot
##############################################
echo "📦 Restoring snapshot on TARGET cluster..."
curl -s -X POST "$TARGET_ES_URL/_snapshot/$REPO_NAME/$SNAPSHOT_NAME/_restore" \
  -H "$TARGET_AUTH_HEADER" \
  -H 'Content-Type: application/json' \
  -d "{
        \"indices\": \"$MATCHING_INDICES\",
        \"rename_pattern\": \"^(.*)\",
        \"rename_replacement\": \"\$1\",
        \"include_global_state\": false
      }"

##############################################
# 📊 Final cluster health check
##############################################
echo "📊 Checking TARGET cluster health post-restore..."
check_cluster_health "$TARGET_ES_URL" "TARGET" "$TARGET_AUTH_HEADER"

##############################################
# 🔓 Optionally re-enable write access on source
##############################################
while true; do
  read -p "Do you want to re-enable write access to source indices? [y/N]: " ENABLE_WRITE
  ENABLE_WRITE=$(echo "$ENABLE_WRITE" | tr '[:upper:]' '[:lower:]')

  if [[ "$ENABLE_WRITE" =~ ^(y|yes)$ ]]; then
    echo "🔓 Re-enabling write access to source indices..."
    for index in $(echo "$MATCHING_INDICES" | tr ',' ' '); do
      curl -s -X PUT "$SOURCE_ES_URL/$index/_settings" \
        -H "$SOURCE_AUTH_HEADER" \
        -H 'Content-Type: application/json' \
        -d '{"settings":{"index.blocks.write":false}}' > /dev/null
    done
    echo "✅ Write access re-enabled on source indices."
    break
  elif [[ -z "$ENABLE_WRITE" || "$ENABLE_WRITE" =~ ^(n|no)$ ]]; then
    echo "ℹ️ Write access to source indices remains blocked (read-only)."
    break
  else
    echo "❌ Invalid input. Please enter 'y' or 'n'."
  fi
done

echo "✅ Migration completed successfully: $SNAPSHOT_NAME"
"
