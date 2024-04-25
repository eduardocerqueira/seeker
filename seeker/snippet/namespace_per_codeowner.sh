#date: 2024-04-25T16:50:26Z
#url: https://api.github.com/gists/03007635dbbb1dda0a33b3de89936a6b
#owner: https://api.github.com/users/SmirnovaNataliia

#!/bin/bash

ROOT_DIR="."

echo -e "Namespace\tCodeowner\tNumber_of_Keys"

# Find all translator directories recursively from the root directory
find "$ROOT_DIR" -type d -name "translator" | while IFS= read -r translator_dir; do
    config_file="$translator_dir/translator.config.env"
    translations_file="$translator_dir/translations/translation.en.json"

    if [[ -f "$config_file" ]]; then
        namespace=$(grep "namespace" "$config_file" | cut -d'=' -f2 | tr -cd '[:alnum:]_-')

        codeowner_path="${translator_dir#./}"  # Strip leading `./`
        team_name=""

        while [[ -z "$team_name" && "$codeowner_path" != "." ]]; do
            codeowner_path=$(dirname "$codeowner_path")

            # Try to find a codeowner for this path without leading `./`
            clean_path="${codeowner_path#./}"  # Ensure no leading `./`
            codeowner_entry=$(grep -m 1 "^/$clean_path" CODEOWNERS)

            if [[ -n "$codeowner_entry" ]]; then
                team_name=$(echo "$codeowner_entry" | grep -o '@Miroapp-dev/.*' | cut -d ' ' -f 1 | cut -d '/' -f 2)
            fi
        done

        # Default value if no codeowner found
        team_name="${team_name:-Unknown}"

        # Count the number of keys in translation.en.json
        num_keys="Unknown"
        if [[ -f "$translations_file" ]]; then
            num_keys=$(jq 'keys | length' "$translations_file")
        fi
        
        # Output a tab-separated row
        echo "$namespace,${team_name:-Unknown},${num_keys:-Unknown}"
    fi
done
