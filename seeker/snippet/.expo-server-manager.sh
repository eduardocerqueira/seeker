#date: 2025-10-21T17:10:19Z
#url: https://api.github.com/gists/690b732de475739678e88dd67c3980da
#owner: https://api.github.com/users/betomoedano

#!/bin/zsh

# Expo Server Manager - Prompts to kill existing port 8081 process before starting Expo dev server
#
# Setup:
# 1. Save this file to ~/.expo-server-manager.sh
# 2. Add to ~/.zshrc: source ~/.expo-server-manager.sh
# 3. Add aliases:
#    alias es='start_expo false'
#    alias esc='start_expo true'
# 4. Restart terminal or run: source ~/.zshrc

# Function to check and handle port 8081
check_port_8081() {
    local pid=$(lsof -ti:8081)
    if [ -n "$pid" ]; then
        echo "Found process $pid running on port 8081"
        echo -n "Stop the existing server and kill the process? (y/n): "
        read answer
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            kill -9 $pid
            echo "✓ Process killed successfully"
            sleep 1
            return 0
        else
            echo "Keeping existing server. Expo will prompt for alternative port..."
            return 1
        fi
    fi
    return 0
}

# Start expo with optional cache clear
start_expo() {
    local clear_cache=$1
    
    check_port_8081
    local should_continue=$?
    
    if [ "$clear_cache" = "true" ]; then
        echo "Starting Expo with cache clear..."
        npx expo start --clear
    else
        echo "Starting Expo..."
        npx expo start
    fi
}
