#date: 2026-03-03T17:36:50Z
#url: https://api.github.com/gists/c116d0432e60d933e6ff54b777e495f9
#owner: https://api.github.com/users/vapvarun

#!/bin/bash
# Check SSL certificate expiry for your domain
curl -vI https://yourdomain.com 2>&1 | grep -E "expire|subject|issuer"

# Cleaner output with date only
echo | openssl s_client -servername yourdomain.com -connect yourdomain.com:443 2>/dev/null \
  | openssl x509 -noout -dates
