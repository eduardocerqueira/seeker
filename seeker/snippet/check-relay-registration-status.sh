#date: 2022-11-17T17:11:30Z
#url: https://api.github.com/gists/73ce799facc2c499065ba140e79135af
#owner: https://api.github.com/users/burz

#!/bin/bash

# The relays you have configured in mev-boost go here
RELAYS=(
  https://0xac6e77dfe25ecd6110b8e780608cce0dab71fdd5ebea22a16c0205200f2f8e2e3ad3b71d3499c54ad14d6c21b41a37ae@boost-relay.flashbots.net
  https://0xb3ee7afcf27f1f1259ac1787876318c6584ee353097a50ed84f51a1f21a323b3736f271a895c7ce918c038e4265918be@relay.edennetwork.io
)

# The public keys for your validators go here
PUBLIC_KEYS=(
  0xPUBLICKEY1
  0xPUBLICKEY2
)

for relay in "${RELAYS[@]}"; do
  echo -e "\n\nChecking $(echo $relay | cut -d '@' -f 2)\n"
  for public_key in "${PUBLIC_KEYS[@]}"; do
    curl "$relay/relay/v1/data/validator_registration?pubkey=$public_key"
  done
done