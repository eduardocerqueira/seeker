#date: 2025-07-15T17:09:03Z
#url: https://api.github.com/gists/534e4bf0856fd165d31b01a9bacddc70
#owner: https://api.github.com/users/aalma4git

#!/bin/bash

echo "=== PoC script executed successfully ==="
echo "Runner user: $(whoami)"
echo "Current directory: $(pwd)"
echo "Environment variables:"
env | grep GITHUB_

echo "PoC executed at $(date)" > /tmp/poc.txt