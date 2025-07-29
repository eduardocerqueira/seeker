#date: 2025-07-29T16:56:46Z
#url: https://api.github.com/gists/f20cf9a251c4c836c96ed13a52144e7b
#owner: https://api.github.com/users/loloop

#!/bin/bash

# Function to display runtimes
display_runtimes() {
    echo -e "\nAvailable iOS Simulator Runtimes:"
    echo "--------------------------------------------------------------------------------"
    
    local counter=1
    while IFS= read -r line; do
        local identifier=$(echo "$line" | jq -r '.key')
        local platform=$(echo "$line" | jq -r '.value.platformIdentifier // "Unknown"' | sed 's/com.apple.platform.//')
        local version=$(echo "$line" | jq -r '.value.version // "Unknown"')
        local build=$(echo "$line" | jq -r '.value.build // "Unknown"')
        local size_bytes=$(echo "$line" | jq -r '.value.sizeBytes // 0')
        local last_used=$(echo "$line" | jq -r '.value.lastUsedAt // "null"')
        
        # Calculate size in GB
        if [ "$size_bytes" -ne 0 ]; then
            local size_gb=$(echo "scale=2; $size_bytes / 1024 / 1024 / 1024" | bc)
            size="${size_gb} GB"
        else
            size="Unknown"
        fi
        
        # Format last used date
        if [ "$last_used" != "null" ]; then
            last_used=$(date -j -f "%Y-%m-%dT%H:%M:%SZ" "$last_used" "+%Y-%m-%d" 2>/dev/null || echo "Unknown")
        else
            last_used="Never"
        fi
        
        echo "$counter. $platform $version"
        echo "   Build: $build"
        echo "   Size: $size"
        echo "   Last used: $last_used"
        echo "   Identifier: $identifier"
        echo
        
        counter=$((counter + 1))
    done
}

# Function to delete runtimes
delete_runtimes() {
    local identifiers=("$@")
    
    echo -e "\nDeleting runtimes..."
    
    for identifier in "${identifiers[@]}"; do
        local runtime_info=$(echo "$RUNTIMES_JSON" | jq -r --arg id "$identifier" '.[$id]')
        local platform=$(echo "$runtime_info" | jq -r '.platformIdentifier // "Unknown"' | sed 's/com.apple.platform.//')
        local version=$(echo "$runtime_info" | jq -r '.version // "Unknown"')
        local display_name="$platform $version"
        
        echo -n "Deleting $display_name..."
        if xcrun simctl runtime delete "$identifier" 2>&1; then
            echo " ✓ Successfully deleted"
        else
            echo " ✗ Failed to delete"
        fi
    done
}

# Main script
echo "iOS Simulator Runtime Deletion Tool"
echo "================================================================================"

# Get simulator runtimes
RUNTIMES_JSON=$(xcrun simctl runtime list -j 2>/dev/null)

if [ $? -ne 0 ]; then
    echo "Error running xcrun simctl"
    exit 1
fi

# Check if there are any runtimes
runtime_count=$(echo "$RUNTIMES_JSON" | jq 'length')

if [ "$runtime_count" -eq 0 ]; then
    echo "No simulator runtimes found."
    exit 0
fi

# Convert JSON to array of entries for easier processing
RUNTIME_ENTRIES=$(echo "$RUNTIMES_JSON" | jq -r 'to_entries[] | @json')

# Display runtimes
echo "$RUNTIME_ENTRIES" | display_runtimes

# Get user selection
echo -e "\nEnter the numbers of the runtimes you want to delete, separated by spaces."
echo "Example: 1 3 5"
echo "Or type \"all\" to select all runtimes"
echo "Press Enter without typing anything to cancel."
echo
read -p "Your selection: " user_input

# Handle empty input (cancel)
if [ -z "$user_input" ]; then
    echo "No runtimes selected. Exiting."
    exit 0
fi

# Build array of selected identifiers
selected_identifiers=()

if [ "$(echo "$user_input" | tr '[:upper:]' '[:lower:]')" = "all" ]; then
    # Select all runtimes
    while IFS= read -r line; do
        identifier=$(echo "$line" | jq -r '.key')
        selected_identifiers+=("$identifier")
    done <<< "$RUNTIME_ENTRIES"
else
    # Parse individual selections
    for num in $user_input; do
        # Validate that it's a number
        if ! [[ "$num" =~ ^[0-9]+$ ]]; then
            echo "Warning: '$num' is not a valid number (ignoring)"
            continue
        fi
        
        # Get the identifier for this selection (1-indexed)
        identifier=$(echo "$RUNTIME_ENTRIES" | sed -n "${num}p" | jq -r '.key' 2>/dev/null)
        
        if [ -n "$identifier" ]; then
            selected_identifiers+=("$identifier")
        else
            echo "Warning: $num is not a valid selection (ignoring)"
        fi
    done
fi

# Check if any runtimes were selected
if [ ${#selected_identifiers[@]} -eq 0 ]; then
    echo "No valid runtimes selected. Exiting."
    exit 0
fi

# Display selected runtimes for confirmation
echo -e "\nYou have selected the following runtimes for deletion:"
echo "--------------------------------------------------------------------------------"

for identifier in "${selected_identifiers[@]}"; do
    runtime_info=$(echo "$RUNTIMES_JSON" | jq -r --arg id "$identifier" '.[$id]')
    platform=$(echo "$runtime_info" | jq -r '.platformIdentifier // "Unknown"' | sed 's/com.apple.platform.//')
    version=$(echo "$runtime_info" | jq -r '.version // "Unknown"')
    build=$(echo "$runtime_info" | jq -r '.build // "Unknown"')
    echo "- $platform $version ($build)"
done

# Confirm deletion
echo -e "\nThis action cannot be undone!"
read -p "Are you sure you want to delete these runtimes? (yes/no): " confirmation

if [ "$(echo "$confirmation" | tr '[:upper:]' '[:lower:]')" = "yes" ]; then
    delete_runtimes "${selected_identifiers[@]}"
    echo -e "\nDeletion complete!"
else
    echo "Deletion cancelled."
fi