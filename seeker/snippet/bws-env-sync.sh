#date: 2026-02-17T17:23:19Z
#url: https://api.github.com/gists/982d7f3b2464fa1ea69eb1e1cfa08ca5
#owner: https://api.github.com/users/adhishthite

#!/usr/bin/env bash
# ============================================================================
# bws-env-sync.sh — Pull BWS secrets into an existing .env, preserving structure
#
# Usage:
#   ./bws-env-sync.sh <.env path> <BWS prefix>
#
# Example:
#   ./bws-env-sync.sh ./apps/analytics/.env ELASTICGPT_ANALYTICS_
#
# Prerequisites:
#   1. Install BWS CLI:
#        brew install bitwarden/tap/bws
#
 "**********"# "**********"  "**********"  "**********"  "**********"2 "**********". "**********"  "**********"S "**********"e "**********"t "**********"  "**********"y "**********"o "**********"u "**********"r "**********"  "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
#        export BWS_ACCESS_TOKEN= "**********"
#      Or add to ~/.zshrc for persistence.
#
#   3. Have a .env file with empty/placeholder values like:
#        # Okta Configuration
#        NEXT_PUBLIC_OKTA_ISSUER_URL=""
#        NEXT_PUBLIC_OKTA_CLIENT_ID=""
#
#        # Elasticsearch Configuration
#        ELASTICSEARCH_URL=""
#        ELASTICSEARCH_API_KEY=""
#        ELASTICSEARCH_INDEX="chats"
#
#        NODE_ENV=development
#
# What happens:
#   - Fetches all BWS secrets whose name starts with your prefix
#   - Walks your .env line by line
#   - If a key matches a BWS secret name, fills in the BWS value
#   - Comments, blank lines, non-matching keys are left untouched
#   - Backs up original to .env.bak, writes filled version to .env
# ============================================================================

set -euo pipefail

# --- Args -------------------------------------------------------------------
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <.env path> <BWS prefix>"
    echo "Example: $0 .env ELASTICGPT_ANALYTICS_"
    exit 1
fi

ENV_FILE="$1"
PREFIX="$2"

# --- Preflight checks -------------------------------------------------------
if [[ ! -f "$ENV_FILE" ]]; then
    echo "❌ File not found: $ENV_FILE"
    exit 1
fi

if ! command -v bws &>/dev/null; then
    echo "❌ bws CLI not found."
    echo "   Install: brew install bitwarden/tap/bws"
    exit 1
fi

if ! command -v jq &>/dev/null; then
    echo "❌ jq not found."
    echo "   Install: brew install jq"
    exit 1
fi

if [[ -z "${BWS_ACCESS_TOKEN: "**********"
    echo "❌ BWS_ACCESS_TOKEN not set."
    echo "   Run: "**********"
    exit 1
fi

# --- Fetch secrets -----------------------------------------------------------
echo "🔑 Fetching BWS secrets with prefix: "**********"

declare -A SECRETS
secret_count= "**********"

while IFS= read -r line; do
    key=$(echo "$line" | jq -r '.key // empty')
    value=$(echo "$line" | jq -r '.value // empty')
    if [[ -n "$key" ]]; then
        SECRETS["$key"]= "**********"
        ((secret_count++))
    fi
done < <(bws secret list --output json 2>/dev/null | jq -c '.[] | select(.key | startswith("'"$PREFIX"'")) | {key: "**********": .value}')

if [[ $secret_count -eq 0 ]]; then
    echo "⚠️  No secrets found with prefix '${PREFIX}'"
    echo "   Check your prefix and BWS_ACCESS_TOKEN."
    echo ""
    echo "   Available prefixes (first 10):"
    bws secret list --output json 2>/dev/null | jq -r '.[].key' | sed 's/_[^_]*$//' | sort -u | head -10
    exit 1
fi

echo "   Found $secret_count secrets."

# --- Backup original ---------------------------------------------------------
BACKUP="${ENV_FILE}.bak"
cp "$ENV_FILE" "$BACKUP"
echo "📋 Backed up: $ENV_FILE → $BACKUP"

# --- Process .env ------------------------------------------------------------
TEMP_FILE=$(mktemp)
filled=0
kept=0
empty_warning=()

while IFS= read -r line || [[ -n "$line" ]]; do
    # Blank lines
    if [[ -z "$line" ]]; then
        echo "" >> "$TEMP_FILE"
        continue
    fi

    # Comments
    if [[ "$line" =~ ^[[:space:]]*# ]]; then
        echo "$line" >> "$TEMP_FILE"
        continue
    fi

    # Key=Value (handles KEY=value, KEY="value", KEY='value', KEY=)
    if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*) ]]; then
        key="${BASH_REMATCH[1]}"
        existing_value="${BASH_REMATCH[2]}"

        if [[ -v "SECRETS[$key]" ]]; then
            bws_value= "**********"
            # Quote the value if it contains spaces or special chars
            if [[ "$bws_value" =~ [[:space:]\#\$\&\|\;\>\<] ]]; then
                echo "${key}=\"${bws_value}\"" >> "$TEMP_FILE"
            else
                echo "${key}=${bws_value}" >> "$TEMP_FILE"
            fi
            ((filled++))
        else
            echo "$line" >> "$TEMP_FILE"
            ((kept++))

            # Warn if value looks empty
            stripped="${existing_value#\"}"
            stripped="${stripped%\"}"
            stripped="${stripped#\'}"
            stripped="${stripped%\'}"
            if [[ -z "$stripped" ]]; then
                empty_warning+=("$key")
            fi
        fi
    else
        # Pass through anything else
        echo "$line" >> "$TEMP_FILE"
    fi
done < "$ENV_FILE"

# --- Write result ------------------------------------------------------------
mv "$TEMP_FILE" "$ENV_FILE"

# --- Summary -----------------------------------------------------------------
echo ""
echo "✅ Done."
echo "   Filled from BWS:  $filled"
echo "   Kept as-is:       $kept"
echo "   Output:           $ENV_FILE"
echo "   Backup:           $BACKUP"

if [[ ${#empty_warning[@]} -gt 0 ]]; then
    echo ""
    echo "⚠️  These keys had empty values and NO BWS match:"
    for k in "${empty_warning[@]}"; do
        echo "   - $k"
    done
    echo "   Add them to BWS with prefix '${PREFIX}' or fill manually."
fi

echo ""
echo "To revert: mv $BACKUP $ENV_FILE"
