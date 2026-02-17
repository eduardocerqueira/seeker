#date: 2026-02-17T17:23:19Z
#url: https://api.github.com/gists/982d7f3b2464fa1ea69eb1e1cfa08ca5
#owner: https://api.github.com/users/adhishthite

#!/usr/bin/env bash
# ============================================================================
# bws-env-push.sh — Push .env secrets INTO Bitwarden Secrets Manager
#
# Usage:
#   ./bws-env-push.sh <.env path> <BWS project-id> [prefix]
#
# Examples:
#   ./bws-env-push.sh .env 9a8b7c6d-1234-5678-abcd-ef0123456789
#   ./bws-env-push.sh .env 9a8b7c6d-1234-5678-abcd-ef0123456789 ELASTICGPT_ANALYTICS_
#
# What it does:
#   1. Reads your .env file (skips comments, blank lines)
#   2. For each KEY= "**********"
#   3. If a secret with that name already exists, updates it
#   4. Dry-run mode by default. Pass --execute to actually push.
#
# Prerequisites:
#   brew install bitwarden/tap/bws jq
#   export BWS_ACCESS_TOKEN= "**********"
# ============================================================================

set -euo pipefail

# --- Args -------------------------------------------------------------------
EXECUTE=false
POSITIONAL=()

for arg in "$@"; do
    case $arg in
        --execute) EXECUTE=true ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done

ENV_FILE="${POSITIONAL[0]:-}"
PROJECT_ID="${POSITIONAL[1]:-}"
PREFIX="${POSITIONAL[2]:-}"

if [[ -z "$ENV_FILE" || -z "$PROJECT_ID" ]]; then
    echo "Usage: $0 <.env path> <BWS project-id> [prefix] [--execute]"
    echo ""
    echo "  --execute    Actually push to BWS (default is dry-run)"
    echo "  prefix       Optional prefix to prepend to each key name"
    echo ""
    echo "Example:"
    echo "  $0 .env 9a8b7c6d-... ELASTICGPT_ANALYTICS_ --execute"
    exit 1
fi

# --- Preflight checks -------------------------------------------------------
if [[ ! -f "$ENV_FILE" ]]; then
    echo "❌ File not found: $ENV_FILE"
    exit 1
fi

if ! command -v bws &>/dev/null; then
    echo "❌ bws CLI not found. Install: brew install bitwarden/tap/bws"
    exit 1
fi

if ! command -v jq &>/dev/null; then
    echo "❌ jq not found. Install: brew install jq"
    exit 1
fi

if [[ -z "${BWS_ACCESS_TOKEN: "**********"
    echo "❌ BWS_ACCESS_TOKEN not set."
    echo "   Run: "**********"
    exit 1
fi

# --- Fetch existing secrets for dedup ----------------------------------------
echo "🔍 Fetching existing BWS secrets..."
declare -A EXISTING_SECRETS

while IFS= read -r line; do
    key=$(echo "$line" | jq -r '.key // empty')
    id=$(echo "$line" | jq -r '.id // empty')
    if [[ -n "$key" && -n "$id" ]]; then
        EXISTING_SECRETS["$key"]= "**********"
    fi
done < <(bws secret list --output json 2>/dev/null | jq -c '.[] | {key: "**********": .id}')

echo "   Found ${#EXISTING_SECRETS[@]} existing secrets."

# --- Parse .env and prepare operations ---------------------------------------
echo ""
if [[ "$EXECUTE" == "false" ]]; then
    echo "📋 DRY RUN (pass --execute to push for real)"
fi
echo ""

created=0
updated=0
skipped=0

while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip blank lines
    [[ -z "$line" ]] && continue

    # Skip comments
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    # Parse KEY=VALUE
    if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*) ]]; then
        key="${BASH_REMATCH[1]}"
        value="${BASH_REMATCH[2]}"

        # Strip surrounding quotes
        if [[ "$value" =~ ^\"(.*)\"$ ]]; then
            value="${BASH_REMATCH[1]}"
        elif [[ "$value" =~ ^\'(.*)\'$ ]]; then
            value="${BASH_REMATCH[1]}"
        fi

        # Skip empty values
        if [[ -z "$value" ]]; then
            echo "   ⏭  ${PREFIX}${key} (empty value, skipping)"
            ((skipped++))
            continue
        fi

        bws_name="${PREFIX}${key}"

        if [[ -v "EXISTING_SECRETS[$bws_name]" ]]; then
            # Update existing
            secret_id= "**********"
            if [[ "$EXECUTE" == "true" ]]; then
                bws secret edit "$secret_id" --value "$value" --output json >/dev/null 2>&1
                echo "   ✏️  ${bws_name} (updated)"
            else
                echo "   ✏️  ${bws_name} → UPDATE (exists, id: "**********":0:8}...)"
            fi
            ((updated++))
        else
            # Create new
            if [[ "$EXECUTE" == "true" ]]; then
                bws secret create "$bws_name" "$value" "$PROJECT_ID" --output json >/dev/null 2>&1
                echo "   ✅ ${bws_name} (created)"
            else
                echo "   ✅ ${bws_name} → CREATE"
            fi
            ((created++))
        fi
    fi
done < "$ENV_FILE"

# --- Summary -----------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ "$EXECUTE" == "true" ]]; then
    echo "✅ Push complete."
else
    echo "📋 Dry run complete. Run with --execute to push."
fi
echo "   Created:  $created"
echo "   Updated:  $updated"
echo "   Skipped:  $skipped (empty values)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [[ "$EXECUTE" == "false" && $((created + updated)) -gt 0 ]]; then
    echo "To push for real:"
    echo "  $0 $ENV_FILE $PROJECT_ID ${PREFIX:+$PREFIX }--execute"
fi
To push for real:"
    echo "  $0 $ENV_FILE $PROJECT_ID ${PREFIX:+$PREFIX }--execute"
fi
