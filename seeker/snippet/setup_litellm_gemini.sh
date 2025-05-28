#date: 2025-05-28T16:57:06Z
#url: https://api.github.com/gists/3251a59cec14079bcf72230d1c83fe6c
#owner: https://api.github.com/users/lazytrot

#!/bin/sh

# Function to safely update or add a key-value pair to a .env file
# Usage: update_env_var "KEY_NAME" "new_value" "path/to/.env_file"
update_env_var() {
    local key="$1"
    local value="$2"
    local env_file="$3"
    local temp_file="${env_file}.tmp"

    # Create .env if it doesn't exist, ensure it's writable by user only
    if [ ! -f "$env_file" ]; then
        touch "$env_file"
        chmod 600 "$env_file"
    fi

    # Escape common sed special characters in the value: &, /, \
    # Note: this is a basic escaping, more complex values might need more care
    local escaped_value=$(echo "$value" | sed 's/[&/\]/\\&/g')

    if grep -q "^${key}=" "$env_file"; then
        # Key exists, replace its value
        sed "s|^${key}=.*|${key}=${escaped_value}|" "$env_file" > "$temp_file" && mv "$temp_file" "$env_file"
    else
        # Key does not exist, append it
        echo "${key}=${escaped_value}" >> "$env_file"
    fi
}

# Function to prompt for a key, showing a preview if it exists
# Usage: manage_api_key "KEY_NAME_IN_ENV" "Descriptive Name" "path/to/.env_file"
# Returns the key value via global variable KEY_RESULT
manage_api_key() {
    local key_env_name="$1"
    local key_desc_name="$2"
    local env_file="$3"
    local current_key_value=""
    KEY_RESULT="" # Reset global result

    if [ -f "$env_file" ]; then
        # Correctly extract value only
        LINE_WITH_KEY=$(grep "^${key_env_name}=" "$env_file")
        if [ -n "$LINE_WITH_KEY" ]; then
            # Use sed to robustly get everything after the first '='
            current_key_value=$(echo "$LINE_WITH_KEY" | sed "s/^${key_env_name}=//")
        fi
    fi

    NEEDS_NEW_INPUT=true
    if [ -n "$current_key_value" ]; then
        KEY_LEN=$(expr length "$current_key_value")
        PREVIEW="********************" # Default for very long keys
        if [ "$KEY_LEN" -gt 7 ]; then
            PREFIX=$(echo "$current_key_value" | cut -c1-4)
            # Suffix: last 3 chars. Awk is more robust for short strings than `cut -c$((KEY_LEN-2))-$KEY_LEN`
            SUFFIX_RAW=$(echo "$current_key_value" | awk '{ L=length($0); if (L >= 3) print substr($0, L-2); else if (L > 0) print substr($0, L); else print ""; }')
            PREVIEW="${PREFIX}...${SUFFIX_RAW}"
        elif [ "$KEY_LEN" -gt 0 ]; then # For keys shorter than 8 chars but not empty
            PREFIX=$(echo "$current_key_value" | cut -c1-1)
            # Suffix: last char if key has more than 1 char
            if [ "$KEY_LEN" -gt 1 ]; then
                SUFFIX_RAW=$(echo "$current_key_value" | awk '{ L=length($0); if (L > 0) print substr($0, L); else print ""; }')
                PREVIEW="${PREFIX}...${SUFFIX_RAW}"
            else
                PREVIEW="${PREFIX}***"
            fi
        fi # No preview for empty current_key_value
        printf "Found existing %s in %s: %s\n" "$key_desc_name" "$env_file" "$PREVIEW"
        printf "Use this key? ([Y]es/[n]o and exit/[r]e-enter): "
        read -r use_existing_choice < /dev/tty # MODIFIED
        case "$use_existing_choice" in
            [Nn]) echo "Exiting. Please update $key_desc_name in $env_file manually if needed and re-run." ; exit 1 ;;
            [Rr]) NEEDS_NEW_INPUT=true ;;
            [Yy]|""|*) NEEDS_NEW_INPUT=false ; KEY_RESULT="$current_key_value" ;; # Default is Yes
        esac
    fi

    if [ "$NEEDS_NEW_INPUT" = true ]; then
        if [ "$key_env_name" = "LITELLM_MASTER_KEY" ] && [ -z "$current_key_value" ] ; then # Only auto-gen if no current master key
             printf "No LiteLLM Master Key found. Generate one? ([Y]es/[n]o to enter manually): "
             read -r gen_master_key < /dev/tty # MODIFIED
             if [ "$gen_master_key" != "n" ] && [ "$gen_master_key" != "N" ]; then # Default to Yes
                if [ -c /dev/urandom ] && type head >/dev/null 2>&1 && type tr >/dev/null 2>&1; then
                    KEY_RESULT="sk-lite-"$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 32)
                    echo "Generated LiteLLM Master Key: $KEY_RESULT"
                else
                    echo "Could not auto-generate LiteLLM Master Key. Please enter one."
                    printf "Enter your %s: " "$key_desc_name"
                    read -r KEY_RESULT < /dev/tty # MODIFIED
                fi
             else
                printf "Enter your %s: " "$key_desc_name"
                read -r KEY_RESULT < /dev/tty # MODIFIED
             fi
        else
            printf "Enter your %s: " "$key_desc_name"
            read -r KEY_RESULT < /dev/tty # MODIFIED
        fi


        if [ -z "$KEY_RESULT" ]; then
            echo "$key_desc_name cannot be empty. Exiting."
            exit 1
        fi
        update_env_var "$key_env_name" "$KEY_RESULT" "$env_file"
        echo "$key_desc_name saved to $env_file."
    fi
}


# --- Main Script ---
echo "Starting LiteLLM Proxy Setup..."

# Check for Docker
if ! type docker >/dev/null 2>&1; then
    echo "Docker is not installed or not in PATH. Please install Docker and try again."
    exit 1
fi

# Navigate to home and create/enter the directory
# Check if HOME is set and not empty
if [ -z "$HOME" ]; then
    echo "ERROR: HOME environment variable is not set. Cannot proceed."
    exit 1
fi
cd "$HOME" || { echo "ERROR: Failed to cd to $HOME"; exit 1; }

mkdir -p litellm-proxy
cd litellm-proxy || { echo "ERROR: Failed to cd to litellm-proxy directory"; exit 1; }

echo "Working directory: $(pwd)"

# Define file names
CONFIG_FILE="litellm-config.yaml"
ENV_FILE=".env"

# --- 1. Handle .env file and API Keys ---
echo "\n--- Configuring API Keys ---"

# Manage GEMINI_API_KEY
manage_api_key "GEMINI_API_KEY" "Gemini API Key" "$ENV_FILE"
GEMINI_API_KEY_FINAL="$KEY_RESULT" # Store the result from the function

# Manage LITELLM_MASTER_KEY (for proxy authentication)
manage_api_key "LITELLM_MASTER_KEY" "LiteLLM Proxy Master Key" "$ENV_FILE"
LITELLM_MASTER_KEY_FINAL="$KEY_RESULT" # Store the result


# --- 2. Create/Update litellm-config.yaml ---
echo "\n--- Creating/Updating $CONFIG_FILE ---"
cat << EOF > "$CONFIG_FILE"
model_list:
  - model_name: gemini-2.5-pro-preview-05-06
    litellm_params:
      model: gemini/gemini-2.5-pro-preview-05-06
      api_key: os.environ/GEMINI_API_KEY # Will use GEMINI_API_KEY from .env
      max_tokens: "**********"
  - model_name: gemini-2.5-flash-preview-05-20
    litellm_params:
      model: gemini/gemini-2.5-flash-preview-05-20
      api_key: os.environ/GEMINI_API_KEY # Will use GEMINI_API_KEY from .env
      max_tokens: "**********"

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY # Will use LITELLM_MASTER_KEY from .env
  # For Admin UI (optional, uncomment and set in .env if desired)
  # litellm_proxy_admin_panel_user: os.environ/LITELLM_ADMIN_USER
  # litellm_proxy_admin_panel_password: "**********"

litellm_settings:
  drop_params: True # Good security practice: prevents users from overriding api_key, model, etc. at request time
  # detailed_debug: True # Uncomment for verbose LiteLLM logs
EOF
echo "$CONFIG_FILE configured successfully."

# --- 3. Docker Operations ---
echo "\n--- Managing Docker Container ---"
CONTAINER_NAME="litellm-proxy"
IMAGE_NAME="ghcr.io/berriai/litellm:main-latest"

# Pull the latest image
echo "Pulling latest LiteLLM image ($IMAGE_NAME)..."
if ! docker pull "$IMAGE_NAME"; then
    echo "Failed to pull Docker image. Please check your internet connection and Docker setup."
    exit 1
fi

# Check container status
if docker ps -q -f name="^/${CONTAINER_NAME}$" | grep -q .; then # Check if output is non-empty
    echo "Container '$CONTAINER_NAME' is already running."
elif docker ps -aq -f status=exited -f name="^/${CONTAINER_NAME}$" | grep -q .; then
    echo "Container '$CONTAINER_NAME' exists but is stopped. Attempting to start..."
    if docker start "$CONTAINER_NAME"; then
        echo "Container '$CONTAINER_NAME' started successfully."
    else
        echo "Failed to start existing container. Removing and recreating..."
        if ! docker rm "$CONTAINER_NAME"; then
            echo "Warning: Failed to remove old container $CONTAINER_NAME. Attempting to proceed may fail."
        fi
        # Will proceed to create new container
    fi
elif docker ps -aq -f name="^/${CONTAINER_NAME}$" | grep -q .; then
    echo "Container '$CONTAINER_NAME' exists in a non-running/non-exited state. Removing and recreating..."
    if ! docker rm "$CONTAINER_NAME"; then
        echo "Warning: Failed to remove old container $CONTAINER_NAME. Attempting to proceed may fail."
    fi
    # Will proceed to create new container
fi

# If container is not running (either never existed, or was stopped and removed), run it
if ! docker ps -q -f name="^/${CONTAINER_NAME}$" | grep -q .; then
    echo "Running new '$CONTAINER_NAME' container..."
    # Note: $(pwd) is used for the volume mount. This assumes the script is run from where $CONFIG_FILE is,
    # which is guaranteed by the cd commands at the start.
    docker run --name "$CONTAINER_NAME" \
        -p 8002:8002 \
        -d \
        --restart always \
        --env-file .env \
        -v "$(pwd)/$CONFIG_FILE:/app/$CONFIG_FILE" \
        "$IMAGE_NAME" \
        --config "/app/$CONFIG_FILE" \
        --port 8002 \
        --host 0.0.0.0

    if [ $? -eq 0 ]; then
        echo "Container '$CONTAINER_NAME' launched successfully."
    else
        echo "ERROR: Failed to start container '$CONTAINER_NAME'."
        echo "Please check Docker daemon and logs (docker logs $CONTAINER_NAME)."
        exit 1
    fi
fi

# --- 4. Final Output ---
echo "\n--- LiteLLM Proxy Setup Complete! ---"
echo "--------------------------------------------------"
echo "Container Name: $CONTAINER_NAME"
# Small delay to allow container to fully start before checking status
sleep 2
if docker ps -q -f name="^/${CONTAINER_NAME}$" | grep -q .; then
    echo "Status:         Running"
else
    echo "Status:         Not Running (check 'docker logs $CONTAINER_NAME')"
fi
echo "Proxy URL:      http://localhost:8002"
echo "Admin Panel:    http://localhost:8002/ui"
echo "  (If you set LITELLM_ADMIN_USER/PASSWORD in .env and uncommented in config)"
echo ""
echo "Client Configuration:"
echo "--------------------------------------------------"
echo "export OPENAI_API_BASE=\"http://localhost:8002\""
echo "export OPENAI_API_KEY=\"${LITELLM_MASTER_KEY_FINAL}\" # This is your LiteLLM Proxy Master Key"
echo ""
echo "Example curl request:"
echo "--------------------------------------------------"
printf "curl -X POST %s/chat/completions \\\\\n" "http://localhost:8002"
printf "  -H \"Content-Type: application/json\" \\\\\n"
printf "  -H \"Authorization: Bearer %s\" \\\\\n" "$LITELLM_MASTER_KEY_FINAL"
printf "  -d '{\n"
printf "    \"model\": \"gemini-2.5-pro-preview-05-06\",\n"
printf "    \"messages\": [{\"role\": \"user\", \"content\": \"Hey, how's it going?\"}]\n"
printf "  }'\n"
echo ""
echo "Troubleshooting & Management:"
echo "--------------------------------------------------"
echo "View logs:    docker logs $CONTAINER_NAME -f"
echo "Stop proxy:   docker stop $CONTAINER_NAME"
echo "Remove proxy: docker rm -f $CONTAINER_NAME"
echo "Restart proxy:docker restart $CONTAINER_NAME"
echo "--------------------------------------------------"--------"