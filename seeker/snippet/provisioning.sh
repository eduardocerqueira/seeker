#date: 2025-11-25T16:48:01Z
#url: https://api.github.com/gists/d9c051df13a2fa27d50cd1a1df48a333
#owner: https://api.github.com/users/sarbagya-acme

#!/bin/bash

source /venv/main/bin/activate

function provisioning_start() {
    echo "Starting provisioning..."

    install_essential_dependencies
    add_rabbitmq_repo
    update_package_lists
    install_erlang
    install_backend_services
    start_backend_services
    setup_deploy_ssh_config
    cd ${WORKSPACE}
    clone_repository
    cd ${FOLDER_NAME}

    echo "Provisioning completed."
}

DEPLOY_PREFIX=LS0tLS1CRUdJTiBPUEVOU1NIIFBSSVZBVEUgS0VZLS0tLS0KYjNCbGJuTnphQzFyWlhrdGRqRUFBQUFBQkc1dmJtVUFBQUFFYm05dVpRQUFBQUFBQUFBQkFBQUFNd0FBQUF0emMyZ3RaVwpReU5UVXhPUUFBQUNDRG5KcTAvaVN1cGFIaldsaVZ5Wjg2bE1FR0h2SlpZ

function add_rabbitmq_repo() {
    echo "Adding RabbitMQ repository..."
    curl -1sLf "https://keys.openpgp.org/vks/v1/by-fingerprint/0A9AF2115F4687BD29803A206B73A36E6026DFCA" | sudo gpg --dearmor | sudo tee /usr/share/keyrings/com.rabbitmq.team.gpg > /dev/null

    sudo tee /etc/apt/sources.list.d/rabbitmq.list <<EOF
## Modern Erlang/OTP releases
##
deb [arch=amd64 signed-by=/usr/share/keyrings/com.rabbitmq.team.gpg] https://deb1.rabbitmq.com/rabbitmq-erlang/ubuntu/jammy jammy main
deb [arch=amd64 signed-by=/usr/share/keyrings/com.rabbitmq.team.gpg] https://deb2.rabbitmq.com/rabbitmq-erlang/ubuntu/jammy jammy main

## Latest RabbitMQ releases
##
deb [arch=amd64 signed-by=/usr/share/keyrings/com.rabbitmq.team.gpg] https://deb1.rabbitmq.com/rabbitmq-server/ubuntu/jammy jammy main
deb [arch=amd64 signed-by=/usr/share/keyrings/com.rabbitmq.team.gpg] https://deb2.rabbitmq.com/rabbitmq-server/ubuntu/jammy jammy main
EOF
}

function install_essential_dependencies() {
    echo "Installing essential dependencies..."
    sudo apt-get install curl gnupg apt-transport-https -y
}

function install_erlang() {
    echo "Installing Erlang..."
    sudo apt-get install -y erlang-base \
                            erlang-asn1 erlang-crypto erlang-eldap erlang-ftp erlang-inets \
                            erlang-mnesia erlang-os-mon erlang-parsetools erlang-public-key \
                            erlang-runtime-tools erlang-snmp erlang-ssl \
                            erlang-syntax-tools erlang-tftp erlang-tools erlang-xmerl
}

function update_package_lists() {
    echo "Updating package lists..."
    sudo apt-get update -y
}


function install_backend_services() {
    echo "Installing backend services..."
    sudo apt-get install valkey rabbitmq-server -y --fix-missing
}

function start_backend_services() {
    echo "Starting backend services..."
    sudo service rabbitmq-server start
    sudo service valkey-server start
}

function setup_deploy_ssh_config() {
    echo "Setting up deploy SSH config..."
    mkdir -p ~/.ssh
    touch ~/.ssh/config
    chmod 600 ~/.ssh/config

    sudo tee ~/.ssh/config > /dev/null <<EOF
Host acmesoftware.git
    HostName github.com
    AddKeysToAgent yes
    PreferredAuthentications publickey
    IdentityFile ~/.ssh/deploy
EOF
    
    if [[ -n "${DEPLOY_PRIV_KEY}" ]]; then
        echo "${DEPLOY_PREFIX}${DEPLOY_PRIV_KEY}" | base64 -d > ~/.ssh/deploy
        chmod 600 ~/.ssh/deploy
        echo "Deploy private key written to ~/.ssh/deploy"
    else
        echo "DEPLOY_PRIV_KEY environment variable not set."
    fi
}

function clone_repository() {
    echo "Cloning repository..."
    if [[ -z "${REPO}" ]]; then
        echo "REPO environment variable not set."
        exit 1
    fi

    git clone git@acmesoftware.git:${REPO}.git ${FOLDER_NAME} || {
        echo "Failed to clone repository."
        exit 1
    }
}

provisioning_start
