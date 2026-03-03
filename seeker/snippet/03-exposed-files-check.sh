#date: 2026-03-03T17:36:50Z
#url: https://api.github.com/gists/c116d0432e60d933e6ff54b777e495f9
#owner: https://api.github.com/users/vapvarun

#!/bin/bash
# Check for exposed sensitive files on your WordPress site
DOMAIN="https://yourdomain.com"

check_file() {
  STATUS=$(curl -o /dev/null -sw "%{http_code}" "$DOMAIN/$1")
  if [ "$STATUS" = "200" ]; then
    echo "EXPOSED ($STATUS): $DOMAIN/$1"
  else
    echo "PROTECTED ($STATUS): $1"
  fi
}

check_file "xmlrpc.php"
check_file "wp-config.php"
check_file "debug.log"
check_file ".env"
check_file "wp-content/debug.log"
check_file "readme.html"
check_file "license.txt"
check_file "wp-admin/install.php"
