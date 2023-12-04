#date: 2023-12-04T17:04:22Z
#url: https://api.github.com/gists/fb79e5aaa2c74b09ec2e5dba25c8e99f
#owner: https://api.github.com/users/adamewozniak

if [ -z "$1" ]; then
  echo "Please provide a moniker"
  exit 1
fi

read -r -p "Are you sure you want to reset your .neutron folder? [y/n]" response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
then
    neutrond tendermint unsafe-reset-all
    rm -r $HOME/.neutron
    neutrond init $1

    content=$(curl https://neutron-rpc.polkachu.com/genesis)
    genesis=$(jq '.result.genesis' <<<"$content")

    rm -r $HOME/.neutron/config/genesis.json
    echo "$genesis" > $HOME/.neutron/config/genesis.json

    # get the state sync block height
    interval=1000
    res=$(curl 'https://neutron-api.polkachu.com/cosmos/base/tendermint/v1beta1/blocks/latest')
    block_hash=$(echo $res | jq -r '.block_id.hash')
    current_block_height=$(echo $res | jq -r '.block.header.height')
    sync_block_height=$(echo "scale=0; (($current_block_height - 3 * $interval) / $interval) * $interval" | bc)

    # get the state sync block hash
    block_query=$(curl 'https://neutron-api.polkachu.com/cosmos/base/tendermint/v1beta1/blocks/'$sync_block_height'')
    hash_base64=$(echo $block_query | jq -r '.block_id.hash')
    # remove any spaces
    hash_hex=$(echo $hash_base64 | base64 -d | xxd -p | tr -d '\n')
    hash_hex=${hash_hex// /}

    SNAP_RPC="https://neutron-rpc.polkachu.com:443/" ; \
    BLOCK_HEIGHT=$sync_block_height; \
    TRUST_HASH=$hash_hex \
    PEERS=$(curl -s https://raw.githubusercontent.com/cosmos/chain-registry/master/neutron/chain.json | jq -r '[foreach .peers.seeds[] as $item (""; "\($item.id)@\($item.address)")] | join(",")')

    sed -i.bak -e "s/^seeds *=.*/seeds = \"$PEERS\"/" $HOME/.neutron/config/config.toml
    sed -i '' 's/enable = false/enable = true/g' $HOME/.neutron/config/config.toml
    sed -i '' "s|rpc_servers = \"\"|rpc_servers = \"$SNAP_RPC,$SNAP_RPC\"|g" $HOME/.neutron/config/config.toml
    sed -i '' "s/trust_height = 0/trust_height = $BLOCK_HEIGHT/g" $HOME/.neutron/config/config.toml
    sed -i '' "s|trust_hash = \"\"|trust_hash = \"$TRUST_HASH\"|g" $HOME/.neutron/config/config.toml
    sed -i '' 's|minimum-gas-prices = ""|minimum-gas-prices = "0.01uneutron"|g' $HOME/.neutron/config/app.toml

    neutrond start
else
    exit 1
fi
