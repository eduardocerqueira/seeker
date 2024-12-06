#date: 2024-12-06T16:50:31Z
#url: https://api.github.com/gists/c583dd9b5b5f83db51f81784f8d7167b
#owner: https://api.github.com/users/iancaseydouglas

#!/bin/bash

# Configuration
CONFIG_DIR="$HOME/.gitflow_config"
SSH_DIR="$HOME/.ssh"

set_defaults() {
    ENVIRONMENT_TYPE=""
    EMAIL=""
    CLIENT_NICKNAME=""
}

print_usage() {
    echo "Usage: $0 --env [personal|client] --email your.email@domain.com --client client-nickname"
    echo
    echo "Sets up GitHub authentication for either personal or client environment"
    echo "Options:"
    echo "  --env     Environment type (personal or client)"
    echo "  --email   Email address for SSH key generation"
    echo "  --client  Short nickname for client company (always required)"
    exit 1
}

generate_ssh_key() {
    local hostname=$1
    local key_file=$2
    local email=$3
    local title=$4
    
    ssh-keygen -t ed25519 -C "$email" -f "$key_file" -N ""
    gh ssh-key add "$key_file.pub" --title "$title"
}

configure_git_signing() {
    local key_file=$1
    git config --global commit.gpgsign true
    git config --global gpg.format ssh
    git config --global user.signingkey "$key_file"
}

create_env_file() {
    local env_file=$1
    local key_file=$2
    local email=$3
    
    cat > "$env_file" << EOF
# Environment setup
export GIT_SSH_COMMAND='ssh -i $key_file'
export GIT_AUTHOR_EMAIL='$email'
export GIT_COMMITTER_EMAIL='$email'
EOF
    chmod 600 "$env_file"
}

setup_personal_environment() {
    local hostname=$(hostname | tr -dc '[:alnum:]\n\r' | tr '[:upper:]' '[:lower:]')
    local key_file="$SSH_DIR/github_${CLIENT_NICKNAME}_${hostname}"
    
    echo "Setting up personal environment for $CLIENT_NICKNAME work on $hostname..."
    
    generate_ssh_key "$hostname" "$key_file" "$EMAIL" "${CLIENT_NICKNAME}-${hostname}"
    configure_git_signing "$key_file"
    create_env_file "$CONFIG_DIR/${CLIENT_NICKNAME}_env.sh" "$key_file" "$EMAIL"
    
    echo "Created environment at $CONFIG_DIR/${CLIENT_NICKNAME}_env.sh"
}

setup_client_environment() {
    local hostname=$(hostname | tr -dc '[:alnum:]\n\r' | tr '[:upper:]' '[:lower:]')
    local key_file="$SSH_DIR/github_${CLIENT_NICKNAME}_${hostname}"
    
    echo "Setting up client environment for $CLIENT_NICKNAME on $hostname..."
    
    generate_ssh_key "$hostname" "$key_file" "$EMAIL" "${CLIENT_NICKNAME}-${hostname}"
    configure_git_signing "$key_file"
    create_env_file "$CONFIG_DIR/${CLIENT_NICKNAME}_env.sh" "$key_file" "$EMAIL"
    
    echo "Setting up client organization access..."
    gh auth login --scopes 'repo,read:org' --hostname github.com
}

create_config_template() {
    mkdir -p "$CONFIG_DIR"
    local config_file="$CONFIG_DIR/${CLIENT_NICKNAME}.yaml"
    
    cat > "$config_file" << EOF
email: "$EMAIL"
ssh_key: "$SSH_DIR/github_${CLIENT_NICKNAME}_$(hostname)"
repositories:
  # Example configuration - edit as needed:
  # api:
  #   personal:
  #     path: /home/user/projects/api
  #     url: git@github.com:username/api.git
  #   client:
  #     path: /client/workspace/api
  #     org: client-org
  #     branch_prefix: feature/sync
EOF
    chmod 600 "$config_file"
    echo "Created configuration template at $config_file"
}

setup_directories() {
    mkdir -p "$SSH_DIR"
    chmod 700 "$SSH_DIR"
    mkdir -p "$CONFIG_DIR"
    chmod 700 "$CONFIG_DIR"
}

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --env)
                ENVIRONMENT_TYPE="$2"
                shift 2
                ;;
            --email)
                EMAIL="$2"
                shift 2
                ;;
            --client)
                CLIENT_NICKNAME="$2"
                shift 2
                ;;
            *)
                print_usage
                ;;
        esac
    done
}

validate_inputs() {
    [ -z "$ENVIRONMENT_TYPE" ] && echo "Error: --env is required" && print_usage
    [ -z "$EMAIL" ] && echo "Error: --email is required" && print_usage
    [ -z "$CLIENT_NICKNAME" ] && echo "Error: --client is required" && print_usage
    
    if [ "$ENVIRONMENT_TYPE" != "personal" ] && [ "$ENVIRONMENT_TYPE" != "client" ]; then
        echo "Error: Environment must be either 'personal' or 'client'"
        exit 1
    fi
}

main() {
    set_defaults
    parse_args "$@"
    validate_inputs
    setup_directories
    
    case "$ENVIRONMENT_TYPE" in
        personal)
            setup_personal_environment
            ;;
        client)
            setup_client_environment
            ;;
    esac
    
    create_config_template
    echo "Setup complete!"
}

main "$@"
