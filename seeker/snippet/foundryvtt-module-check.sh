#date: 2024-06-03T17:06:05Z
#url: https://api.github.com/gists/d4493860f769c6d1c2febdee0906d44c
#owner: https://api.github.com/users/cs96and

#!/bin/sh

# This script checks the compatibility of all modules in the modules directory with the specified FoundryVTT version.

# Constants
MODULES_DIR="/data/Data/modules"
TARGET_MAJOR_VERSION="12"

log "Checking module compatibility with FoundryVTT version $TARGET_MAJOR_VERSION"

check_compatibility() {
    local manifest_url=$1
    local latest_module_json=$(curl -sL "$manifest_url")

    if [ -n "$latest_module_json" ]; then
        local latest_title=$(echo "$latest_module_json" | jq -r '.title')
        local latest_version=$(echo "$latest_module_json" | jq -r '.version')
        local verified_version=$(echo "$latest_module_json" | jq -r '.compatibility.verified')

        if echo "$verified_version" | grep -Eq "^${TARGET_MAJOR_VERSION}(\.|$)"; then
            log "$latest_title ($latest_version), Verified: $verified_version"
        else
            log_warn "$latest_title ($latest_version), NOT Verified: $verified_version"
        fi
    else
        log_error "Failed to fetch manifest from $manifest_url"
    fi
}

for module_dir in "$MODULES_DIR"/*; do
    if [ -d "$module_dir" ]; then
        module_json="$module_dir/module.json"

        if [ -f "$module_json" ]; then
            manifest_url=$(jq -r '.manifest' "$module_json")

            if [ -n "$manifest_url" ]; then
                check_compatibility($manifest_url) &
            else
                log_error "Manifest URL not found in $module_json"
            fi
        fi
    fi
done
