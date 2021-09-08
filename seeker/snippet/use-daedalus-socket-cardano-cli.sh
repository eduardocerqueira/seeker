#date: 2021-09-08T17:12:29Z
#url: https://api.github.com/gists/69df40b6dd8f461fec9f551523b6a9c4
#owner: https://api.github.com/users/cnftexchange

#!/usr/bin/env bash

# Install cardano-cli or use docker https://gist.github.com/ilyar/bf4c2346be1a74c50e488181986808fb
#
# Linux https://hydra.iohk.io/job/Cardano/cardano-node/cardano-node-linux/latest-finished
# Win64 https://hydra.iohk.io/job/Cardano/cardano-node/cardano-node-win64/latest-finished
# Macos https://hydra.iohk.io/job/Cardano/cardano-node/cardano-node-macos/latest-finished
# Extcact only cardano-cli into /usr/local/bin/cardano-cli
# Check
cardano-cli --version

#########################################
# Daedalus Wallet for the Cardano Testnet
# Download https://developers.cardano.org/en/testnets/cardano/get-started/wallet/
# Run Daedalus for Testnet

# Create var CARDANO_NODE_SOCKET_PATH
export CARDANO_NODE_SOCKET_PATH=$(ps ax | grep -v grep | grep cardano-wallet | grep testnet | sed -E 's/(.*)node-socket //')

# Check var it must be path for file of node socket and not empty
echo $CARDANO_NODE_SOCKET_PATH

# Check connect if yor run Daedalus for Testnet
cardano-cli get-tip --testnet-magic 1097911063

#########################################
# Daedalus Wallet for the Cardano Mainnet
# Download https://daedaluswallet.io/en/download/
# Run Daedalus for Mainnet

# Create var CARDANO_NODE_SOCKET_PATH
export CARDANO_NODE_SOCKET_PATH=$(ps ax | grep -v grep | grep cardano-wallet | grep mainnet | sed -E 's/(.*)node-socket //')

# Check var it must be path for file of node socket and not empty
echo $CARDANO_NODE_SOCKET_PATH

# Check connect if yor run Daedalus for Mainnet
cardano-cli get-tip --mainnet
