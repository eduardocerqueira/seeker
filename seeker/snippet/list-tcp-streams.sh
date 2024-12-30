#date: 2024-12-30T17:03:46Z
#url: https://api.github.com/gists/35042b28c1a50398e4be844c2cc96238
#owner: https://api.github.com/users/marshalljmiller

#!/bin/sh

PCAP_FILE=foo.pcap

tshark -r "$PCAP_FILE" -e tcp.stream -e ip.addr -e tcp.port | sort -un