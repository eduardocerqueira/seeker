#date: 2024-03-07T18:18:39Z
#url: https://api.github.com/gists/0b627fea5340bb57b5ffbb4c22d196d3
#owner: https://api.github.com/users/rootulp

#!/bin/sh

# Stop script execution if an error is encountered
set -o errexit
# Stop script execution if an undefined variable is used
set -o nounset

# Prerequisite: prior to running this script, start a single node devnet with ./scripts/single-node.sh
CHAIN_ID="private"
KEY_NAME="validator"
KEYRING_BACKEND="test"
BROADCAST_MODE="block"

celestia-appd keys add test1
celestia-appd keys add test2
celestia-appd keys add test3
celestia-appd keys add test4
celestia-appd keys add multisig --multisig test1,test2,test3 --multisig-threshold 2

TEST1=$(celestia-appd keys show test1 -a)
TEST2=$(celestia-appd keys show test2 -a)
TEST3=$(celestia-appd keys show test3 -a)
TEST4=$(celestia-appd keys show test4 -a)
MULTISIG=$(celestia-appd keys show multisig -a)
VALIDATOR=$(celestia-appd keys show validator -a)

celestia-appd tx bank send $VALIDATOR $MULTISIG 100000utia --from $VALIDATOR --fees 1000utia --chain-id $CHAIN_ID --keyring-backend $KEYRING_BACKEND --broadcast-mode $BROADCAST_MODE --yes
# Uncomment the next line to verify that the multisig account received some funds
celestia-appd query bank balances $MULTISIG

celestia-appd tx bank send $MULTISIG $VALIDATOR 1utia --from $MULTISIG --fees 1000utia --chain-id $CHAIN_ID --keyring-backend $KEYRING_BACKEND --generate-only > unsignedTx.json
celestia-appd tx sign unsignedTx.json --multisig $MULTISIG --from test1 --output-document test1sig.json --chain-id $CHAIN_ID
celestia-appd tx sign unsignedTx.json --multisig $MULTISIG --from test2 --output-document test2sig.json --chain-id $CHAIN_ID
# Uncomment the next lines to verify that the signatures were created
# cat test1sig.json | jq .
# cat test2sig.json | jq .
# celestia-appd tx multisign unsignedTx.json multisig test1sig.json test2sig.json --output-document=signedTx.json --chain-id $CHAIN_ID --generate-only

# The following command should error out because test4 is not part of the multisig account
celestia-appd tx sign unsignedTx.json --multisig $MULTISIG --from test4 --output-document test4sig.json --chain-id $CHAIN_ID
