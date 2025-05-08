#date: 2025-05-08T16:56:05Z
#url: https://api.github.com/gists/9b24e2a31a2fb8a0973ce92e86ae5d7a
#owner: https://api.github.com/users/clemensgg

#!/usr/bin/env bash
set -euo pipefail

# LCD base URL (first arg) or default to localhost
LCD="${1:-https://babylon.nodes.guru/api}"
PAGE_LIMIT=100

# Fetch total bonded tokens
TOTAL=$(curl -s "$LCD/cosmos/staking/v1beta1/pool" \
  | jq -r '.pool.bonded_tokens')

# Prepare a temp file and ensure cleanup
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

NEXT_KEY=""
while :; do
  URL="$LCD/cosmos/staking/v1beta1/validators?status=BOND_STATUS_BONDED&pagination.limit=$PAGE_LIMIT"
  [[ -n "$NEXT_KEY" ]] && URL+="&pagination.key=$NEXT_KEY"

  RESP=$(curl -s "$URL")

  # === OPTION A: no quoting (fastest, but unsafe if fields contain commas/newlines) ===
  echo "$RESP" | jq -r --arg total "$TOTAL" '
    .validators[]
    | [
        .description.moniker,
        .operator_address,
        .tokens,
        ((.tokens | tonumber) / ($total | tonumber) * 100)
      ]
    | @tsv
  ' | sed $'s/\t/,/g' >> "$TMP"

  NEXT_KEY=$(jq -r '.pagination.next_key' <<<"$RESP")
  [[ "$NEXT_KEY" == "null" || -z "$NEXT_KEY" ]] && break
done

# Output header + sorted rows
echo "moniker,validator_address,voting_power,voting_power_percent"
sort -t, -k3,3nr "$TMP"
