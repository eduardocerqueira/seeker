#date: 2026-03-03T17:36:50Z
#url: https://api.github.com/gists/c116d0432e60d933e6ff54b777e495f9
#owner: https://api.github.com/users/vapvarun

#!/bin/bash
# Fetch HTTP response headers and check for security headers
curl -sI https://yourdomain.com | grep -iE \
  "strict-transport|content-security|x-frame|x-content-type|referrer-policy|permissions-policy"

# Quick pass/fail summary
echo ""
echo "=== Security Header Check ==="
HEADERS=$(curl -sI https://yourdomain.com)

check_header() {
  echo "$HEADERS" | grep -qi "$1" && echo "PASS: $1" || echo "FAIL: $1"
}

check_header "Strict-Transport-Security"
check_header "X-Frame-Options"
check_header "X-Content-Type-Options"
check_header "Content-Security-Policy"
check_header "Referrer-Policy"
