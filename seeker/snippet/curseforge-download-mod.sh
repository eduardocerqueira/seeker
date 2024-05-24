#date: 2024-05-24T17:02:59Z
#url: https://api.github.com/gists/581afd4b56eb6239c4fc182c5a82b3ff
#owner: https://api.github.com/users/zekroTJA

#!/bin/bash

mod_id="$1"
version="$2"

get_file_id() {
    local loader="$1"
    local version="$2"

    curl -Ls "https://www.curseforge.com/api/v1/mods/${mod_id}/files?pageIndex=0&pageSize=20&sort=dateCreated&sortDescending=true&removeAlphas=true" \
        | jq -r '[ .data[] | select( .gameVersions | any( . == "'"$loader"'" ) and any(. == "'"$version"'") ) ][0].id'
}


file_id=$(get_file_id "NeoForge" "$version")
if [ "$file_id" == "null" ]; then
    file_id=$(get_file_id "Forge" "$version")
fi

file_name=$(curl -Ls "https://www.curseforge.com/api/v1/mods/${mod_id}/files/${file_id}" \
    | jq -r '.data.fileName')

curl -Lso "$file_name" "https://www.curseforge.com/api/v1/mods/${mod_id}/files/${file_id}/download"

echo "$file_name"