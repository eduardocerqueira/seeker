#date: 2023-08-25T17:08:25Z
#url: https://api.github.com/gists/a8761a820439b37ef9ed818e32e04f23
#owner: https://api.github.com/users/csknk

#!/usr/bin/env bash
# Basic script to run against Bitcoin bitcoind server on LAN
# Ref: Bitcoin Core getchaintips RPC command: https://bitcoincore.org/en/doc/0.21.0/rpc/blockchain/getchaintips

set -eou pipefail
IFS=$'\n\t'
# Set variables ----
user=XXXXXX
password= "**********"
node_ip=192.168.0.XXX
# End --------------

port=8332

data=$(
	cat <<-EOF
		{
			"jsonrpc": "1.0",
			"id": "curltest",
			"method": "getchaintips"
		}
	EOF
)

# Add -vvvv for verbose output/debugging
curl \
	-vvvv \
	--user "${user}: "**********"
	--data-binary "$data" \
	-H 'content-type: text/plain;' \
	"${node_ip}:${port}"
\
	"${node_ip}:${port}"
