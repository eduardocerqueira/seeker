#date: 2026-03-03T17:36:50Z
#url: https://api.github.com/gists/c116d0432e60d933e6ff54b777e495f9
#owner: https://api.github.com/users/vapvarun

#!/bin/bash
# Extract all external JavaScript sources from your WordPress homepage
# Requires: curl, grep, sed

DOMAIN="https://yourdomain.com"

echo "=== External JS Sources on $DOMAIN ==="
curl -s "$DOMAIN" | grep -oP '(?<=src=["\'])[^"\']+\.js[^"\']*' | \
  grep -v "$DOMAIN" | \
  grep -v "^/" | \
  sort -u

echo ""
echo "=== Inline Script Blocks (potential injection risk) ==="
curl -s "$DOMAIN" | grep -c "<script>" && echo "inline script blocks found"
