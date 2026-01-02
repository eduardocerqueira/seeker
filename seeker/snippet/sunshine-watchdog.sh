#date: 2026-01-02T17:11:59Z
#url: https://api.github.com/gists/d6a6cb2b1b45bf18ab626c5be2ac60d6
#owner: https://api.github.com/users/KaiStarkk

#!/usr/bin/env bash
# Watches Sunshine log for disconnects and force-closes session via API
# This triggers undo prep commands that would otherwise hang in "paused" state

SUNSHINE_LOG="${XDG_CONFIG_HOME:-$HOME/.config}/sunshine/sunshine.log"
SUNSHINE_API="https://localhost:47990/api"
SUNSHINE_USER="${SUNSHINE_USER:-your-username}"
SUNSHINE_PASS_FILE="${SUNSHINE_PASS_FILE: "**********"

[[ ! -f "$SUNSHINE_LOG" ]] && { echo "Waiting for sunshine.log..."; while [[ ! -f "$SUNSHINE_LOG" ]]; do sleep 5; done; }

tail -n0 -F "$SUNSHINE_LOG" 2>/dev/null | while read -r line; do
  if [[ "$line" == *"CLIENT DISCONNECTED"* ]]; then
    curl -sk -X POST -H "Content-Type: application/json" \
      -u "$SUNSHINE_USER:$(cat "$SUNSHINE_PASS_FILE")" \
      "$SUNSHINE_API/apps/close"
  fi
done
pps/close"
  fi
done
