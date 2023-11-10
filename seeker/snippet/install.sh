#date: 2023-11-10T16:59:22Z
#url: https://api.github.com/gists/95e66852ea849d38d3e512274a1d95ff
#owner: https://api.github.com/users/khadherinc

#!/bin/bash
export DEBIAN_FRONTEND=noninteractive

read -p "Do you have a SESSION_ID scanned today? (y|n): " IS_SESSION_ID
if [[ "$IS_SESSION_ID" == "y" ]]; then
    read -p "Enter Your SESSION_ID: " SESSION_ID
elif [[ "$IS_SESSION_ID" == "n" ]]; then
    read -p "Do you want to continue without SESSION_ID? You can scan QR from this terminal on starting. (y|n): " IS_CONT
    [ "$IS_CONT" == "y" ] || exit 0
else
    exit 0
fi

read -p "Enter a name for BOT (e.g., levanter): " BOT_NAME
BOT_NAME=${BOT_NAME:-levanter}

echo "Updating system packages..."
sudo apt update -y

for pkg in git ffmpeg curl; do
    if ! [ -x "$(command -v $pkg)" ]; then
        echo "Installing $pkg..."
        sudo apt -y install $pkg
    fi
done

if ! [ -x "$(command -v node)" ] || [[ "$(node -v | cut -c 2-)" -lt 16 ]]; then
    echo "Installing Node.js..."
    sudo apt-get purge nodejs
    rm -rf /etc/apt/sources.list.d/nodesource.list
    rm -rf /etc/apt/keyrings/nodesource.gpg
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | sudo tee /etc/apt/sources.list.d/nodesource.list
    sudo apt-get update
    sudo apt-get install nodejs -y
fi

if ! [ -x "$(command -v yarn)" ]; then
    echo "Installing Yarn..."
    sudo npm install -g yarn
fi

if ! [ -x "$(command -v pm2)" ]; then
    echo "Installing PM2..."
    sudo yarn global add pm2
fi

echo "Installing Levanter..."
git clone https://github.com/khadherinc/001 "$BOT_NAME"
cd "$BOT_NAME" || exit 1
yarn install --network-concurrency 1

echo "Creating config.env file..."
cat >config.env <<EOL
PREFIX=.
STICKER_PACKNAME=LyFE
ALWAYS_ONLINE=false
RMBG_KEY=null
LANGUAG=en
WARN_LIMIT=3
FORCE_LOGOUT=false
BRAINSHOP=159501,6pq8dPiYt7PdqHz3
MAX_UPLOAD=60
REJECT_CALL=false
SUDO=989876543210
TZ=Asia/Kolkata
VPS=true
AUTO_STATUS_VIEW=true
SEND_READ=true
AJOIN=true
EOL
[ "$SESSION_ID" != "1" ] && echo "SESSION_ID=$SESSION_ID" >>config.env

echo "Starting the bot..."
pm2 start index.js --name "$BOT_NAME" --attach --time