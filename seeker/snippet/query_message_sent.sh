#date: 2025-04-28T17:06:11Z
#url: https://api.github.com/gists/7cd45d0a59846d181dedf873e0b6f663
#owner: https://api.github.com/users/carte7000

#!/usr/bin/env bash
# query_message_sent.sh
#
# Prints the latest CCIPMessageSent event emitted by an OnRamp contract
# for a given destChainSelector, even on very old Foundry releases.
#
# Usage:
#   ./query_message_sent.sh <ONRAMP_ADDRESS> <DEST_CHAIN_SELECTOR_DEC> [RPC_URL]
#
# Prerequisites: foundry-cast, jq

set -euo pipefail

###############################################################################
# 1. Parse CLI args
###############################################################################
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <ONRAMP_ADDRESS> <DEST_CHAIN_SELECTOR_DEC> [RPC_URL]" >&2
  exit 1
fi
RAW_ADDR="$1"                 # on-chain OnRamp address (any checksum style)
DEST_CHAIN_DEC="$2"           # decimal uint64 destChainSelector
RPC_URL=${3:-${ETH_RPC_URL:-}}

[[ -n "$RPC_URL" ]] || { echo "❌  RPC URL missing" >&2; exit 1; }

###############################################################################
# 2. Normalise the OnRamp address (best effort for old/new cast)
###############################################################################
if cast --help 2>&1 | grep -q "to-checksum-address"; then
  ONRAMP_ADDR="$(cast to-check-sum-address "$RAW_ADDR")"
else
  ONRAMP_ADDR="$RAW_ADDR"       # fall back to whatever was passed
fi

###############################################################################
# 4. Event signature (topic-0) — must match exactly
###############################################################################
EVENT_SIG='CCIPMessageSent(uint64,uint64,((bytes32,uint64,uint64,uint64,uint64),address,bytes,bytes,bytes,address,uint256,uint256,(address,bytes,bytes,uint256,bytes)[]))'

echo "🔍  Fetching latest CCIPMessageSent for destChainSelector=$DEST_CHAIN_DEC" >&2

###############################################################################
# 5. Call cast logs
#    Your Foundry version parses:   cast logs [OPTIONS] SIGNATURE TOPIC1 ...
###############################################################################

# Fetch all logs matching the event signature and the specific destination chain selector topic
RAW=$(cast logs --rpc-url "$RPC_URL" \
          --address "$ONRAMP_ADDR" \
          --from-block 8123346 --to-block latest \
          --json \
          "$EVENT_SIG") # Filter by destChainSelector topic

# Check if the resulting JSON array is empty
# Use jq -e to exit with non-zero status if the array is empty or null
if ! jq -e '. | length > 0' <<<"$RAW" >/dev/null; then
  echo "⚠️  no logs found for destChainSelector=$DEST_CHAIN_DEC" >&2
  exit 0
fi

EVENT_SIG='CCIPMessageSent(uint64,uint64,(bytes32,uint64,uint64,uint64,uint64,address,bytes,bytes,bytes,address,uint256,uint256,(address,address,bytes,uint256,bytes)[]))'

echo "--- Found $(jq '. | length' <<<"$RAW") matching log(s) ---" >&2
jq -c '.[]' <<<"$RAW" | while IFS= read -r log_json; do
  DATA=$(jq -r '.data'          <<<"$log_json")

  

  T1=$(  jq -r '.topics[1]'     <<<"$log_json")   # destChainSelector
  T2=$(  jq -r '.topics[2]'     <<<"$log_json")   # sequenceNumber

    # Convert T2 (hex) to decimal for the condition check
    DEC_T2=$((T2))

    # Skip if sequence number is not 1 or 3734
    if [[ "$DEC_T2" -ne 1 && "$DEC_T2" -ne 3734 ]]; then
        continue
    fi

  echo $T1
  printf " SequenceNumber: %d\n" $T2
  echo "Data: $DATA"

done
