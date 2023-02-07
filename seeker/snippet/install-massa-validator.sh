#date: 2023-02-07T16:57:52Z
#url: https://api.github.com/gists/bfe04c64751e77730fc399757879f812
#owner: https://api.github.com/users/tranghaviet

#!/bin/bash
sudo apt install jq -y

APP_VERSION=$(curl -s https://api.github.com/repos/massalabs/massa/releases/latest | jq -r ".tag_name")
echo "Newest version: $APP_VERSION"

LINK="https://github.com/massalabs/massa/releases/download/${APP_VERSION}/massa_${APP_VERSION}_release_linux.tar.gz"
wget -c "$LINK" -O - | tar -xz -C ./

echo  "**********"Enter staking keys password: "**********"
read -e staking_password

echo "$staking_password" > ~/massa/massa-node/password.txt
chmod 600 ~/massa/massa-node/password.txt

echo  "**********"Enter wallet client password: "**********"
read -e wallet_password

echo "$wallet_password" > ~/massa/massa-client/password.txt
chmod 600 ~/massa/massa-client/password.txt

echo "[Unit]
Description=Massa Test Node
StartLimitInterval=350
StartLimitBurst=10

[Service]
# Limit CPU usage if you want
# CPUWeight=20
# CPUQuota=25%
# IOWeight=20
# MemorySwapMax=0

User=$USER
WorkingDirectory=$HOME/massa/massa-node
# ExecStart= "**********"=\$(cat \$HOME/massa/massa-node/password.txt)'
ExecStart= "**********"=$staking_password
KillSignal=SIGINT
Restart=on-failure
RestartSec=60
[Install]
WantedBy=multi-user.target" > massa-test-node.service

sudo mv massa-test-node.service /etc/systemd/system/

echo "Config routability"
echo "
[network]
routable_ip = \"$(curl -q ifconfig.me)\"
" > ~/massa/massa-node/config/config.toml

echo "Starting service..."
sudo systemctl daemon-reload
sudo systemctl enable massa-test-node.service
sudo systemctl restart massa-test-node

echo "Done."
echo "You can check service log by running:"
echo "journalctl -u massa-test-node -f"

echo "Next step:"

echo "1. Create wallet: "
echo "cd ~/massa/massa-client"
echo "./massa-client --pwd= "**********"
echo "Type: "**********"
echo "Show wallet: wallet_info"
echo "Type: "**********"

echo "2. Staking"
echo "Paste wallet address to #testnet-faucet https://discord.com/channels/828270821042159636/866190913030193172"
echo "Wait 1 minute"
echo "Enter: buy_rolls <wallet_address> 1 0"
# echo "Wait 1 minute"
# echo "Type: "**********"
echo "Wait 3 cycles of 128 periods ~ 1h40 minutes"
echo "Type wallet_info should show: Active / Final rolls: 1"

echo "3. Testnet Staking Rewards Program"
echo "React to get discord_id from Massa Bot: https://discord.com/channels/828270821042159636/872395473493839913"
echo "Client command: node_testnet_rewards_program_ownership_proof wallet_address discord_id"
echo "Then send output to Massa Bot"
echo "Final step: send $(curl -q ifconfig.me) to Massa Bot"
program_ownership_proof wallet_address discord_id"
echo "Then send output to Massa Bot"
echo "Final step: send $(curl -q ifconfig.me) to Massa Bot"
