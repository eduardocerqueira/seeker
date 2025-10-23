#date: 2025-10-23T17:13:25Z
#url: https://api.github.com/gists/b51b7e0738017a6fc3ad0585eec3af9f
#owner: https://api.github.com/users/JeremyVV

load_config() {
    local config_file="$1"
    
    # Check if the file exists and is readable
    if [[ ! -r "$config_file" ]]; then
        echo "Error: Configuration file $config_file not found or unreadable." >&2
        return 1
    fi

    # NOTE: This function only reads simple KEY=value pairs. 
    # It does not support shell arrays (KEY=(v1 v2)) or associative arrays (hashes).

    while IFS= read -r line; do
        # Clean up leading/trailing whitespace on the whole line
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]*}"}"
        
        # Skip empty lines or lines starting with '#' (comments)
        if [[ -z "$line" || "$line" =~ ^# ]]; then
            continue
        fi

        # Split key and value on the first '=' occurrence
        key="${line%%=*}"
        value="${line#*=}"

        # Clean up whitespace on the key and value
        key="${key#"${key%%[![:space:]]*}"}"
        value="${value#"${value%%[![:space:]]*}"}" 
        
        # Verify the key is a valid Bash variable name
        if ! [[ "$key" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
            echo "Warning: Invalid variable name '$key'. Skipping." >&2
            continue
        fi

        # Safely quote the value to prevent command injection
        local safe_value
        printf -v safe_value '%q' "$value"
        
        # Assign the variable in the calling script's scope
        eval "$key=$safe_value"

    done < "$config_file"
}