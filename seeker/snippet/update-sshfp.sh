#date: 2023-05-15T16:53:09Z
#url: https://api.github.com/gists/6a84139173a6f098b6f69c7353d28c73
#owner: https://api.github.com/users/Zash

#!/bin/bash

set -eo pipefail

ZONE="$(hostname -d)"
FQDN="$(hostname -f)"
UPDATES="$(mktemp --suffix .nsupdate)"
trap 'rm -- "$UPDATES"' EXIT

{
echo "server $(dig +noall +short "$ZONE" soa | cut -d' ' -f1)"
echo "zone $ZONE"
echo "ttl 3600"
echo "del $FQDN IN SSHFP"
ssh-keygen -r "$FQDN" | sed 's/^/add /'
echo "show"
echo "send"
echo "answer"
} > "$UPDATES"
if [ -f "$HOME/.config/nsupdate/$ZONE.key" ]; then
	nsupdate -k "$HOME/.config/nsupdate/$ZONE.key" "$UPDATES"
else
	cat "$UPDATES"
fi
