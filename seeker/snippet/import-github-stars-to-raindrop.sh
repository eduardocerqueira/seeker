#date: 2023-03-28T16:49:50Z
#url: https://api.github.com/gists/47d45f7dd1a5f85ca5634eeb4aba1b2b
#owner: https://api.github.com/users/ScottJWalter

#!/bin/bash
GITHUB_USER="..."
GITHUB_TMP_FILE="$HOME/github-stars.json"
RAINDROP_API_TOKEN= "**********"
GITHUB_TOKEN= "**********"

GITHUB_API_HEADER_ACCEPT="Accept: application/vnd.github.star+json" # "Accept: application/vnd.github.v3+json"
GITHUB_API_VERSION="X-GitHub-Api-Version: 2022-11-28"

sanitize_string() {
    output_string="$1"
    if [[ "$output_string" =~ .*"null,".* ]]; then # if input is null, return empty string
        echo -e "\"\"\n"
    else
        output_string="$(echo $output_string | sed -e 's/^[ \t]*//')" # remove leading spaces
        output_string="$(echo ${output_string:1})"                    # remove first char (")
        output_string="$(echo ${output_string:0:-2})"                 # remove last 2 chars (",)
        output_string="${output_string//\\/}"                         # remove escape char
        output_string="${output_string//\"/\\\"}"                     # convert quotes to escaped quote char
        output_string="$(echo -e "\"$output_string\"")"
        echo "$output_string"
    fi
}

sanitize_date() {
    output_string="$1"
    if [[ "$output_string" =~ .*"null,".* ]]; then # if input is null, return empty string
        echo -e "\"\"\n"
    else
        output_string="$(echo $output_string | sed -e 's/^[ \t]*//')" # remove leading spaces
        output_string="$(echo ${output_string:1})"                    # remove first char (")
        output_string="$(echo ${output_string:0:-2})"                 # remove last 2 chars (",)
        output_string="${output_string//\\/}"                         # remove escape char
        output_string="${output_string//\"/\\\"}"                     # convert quotes to escaped quote char
        output_string="$(echo -e "\"${output_string}Z\"")"
        echo "$output_string"
    fi
}

post_to_raindrop() {
    JSON="${1:0:-1}"
    JSON="$(echo '{ "items": [ '"$JSON"' ] }' | jq -rc '.')"
    curl -s -X POST -H 'Content-Type: "**********": Bearer $RAINDROP_API_TOKEN" -d "$JSON" https://api.raindrop.io/rest/v1/raindrops >/dev/null
}

echo "Getting github stars"
if [ -f "$GITHUB_TMP_FILE" ]; then
    rm "$GITHUB_TMP_FILE" 2>/dev/null
fi

PAGE=1
while curl -s -H "Authorization: "**********"
    -H "${GITHUB_API_VERSION}" \
    -H "${GITHUB_API_HEADER_ACCEPT}" \
    "https://api.github.com/users/$GITHUB_USER/starred?per_page=100&page=$PAGE" |
    jq -r -e '.[] | (.repo | [.name,.html_url,.description,.language])+([.starred_at])' >>"$GITHUB_TMP_FILE"; do
    printf "."
    ((PAGE++))
done
printf "\n"

echo "Getting raindrop collection id"
COLLECTION_ID=$(curl -s -H "Authorization: "**********"://api.raindrop.io/rest/v1/collections | jq -rc '.items[] | select(.title=="Github") ._id')

echo "Removing old items from collection"
curl -s -X DELETE -H "Authorization: "**********"://api.raindrop.io/rest/v1/raindrops/$COLLECTION_ID" >/dev/null
echo "Emptying the trash"
curl -s -X DELETE -H "Authorization: "**********"://api.raindrop.io/rest/v1/raindrops/-99" >/dev/null

echo "Posting to raindrop"
process_counter=0
while mapfile -t -n 7 ary && ((${#ary[@]})); do # read every 7 lines (the size of the json object)

    # sanitize the data
    title=$(sanitize_string "${ary[1]}")
    link=$(sanitize_string "${ary[2]}")
    description=$(sanitize_string "${ary[3]}")
    language=$(sanitize_string "${ary[4]}")
    created=$(sanitize_date "${ary[5]}")

    # set tags and add extra "imported" tag
    extra_tag="\"imported\""
    tags="${extra_tag}${language:+,$language}"

    # create json payload
    JSON="$JSON $(printf '{ "title":%s, "excerpt":%s, "tags": [%s], "link":%s, "created":%s, "lastUpdate":%s, "collection": { "$ref": "collections", "$id": "%s", "oid": "-1" } },' "$title" "$description" "$tags" "$link" "$created" "$created" "$COLLECTION_ID")"
    #echo "$JSON"
    ((process_counter++))

    # batch the payload into 100 items
    if [[ $process_counter -eq 100 ]]; then
        printf "."
        post_to_raindrop "$JSON"
        JSON=""
        process_counter=0
    fi
done <"$GITHUB_TMP_FILE"

if [[ $process_counter -gt 0 ]]; then
    printf "."
    post_to_raindrop "$JSON"
    JSON=""
fi

printf "\n"

rm "$GITHUB_TMP_FILE" 2>/dev/null

echo "Done!"
"$JSON"
    JSON=""
fi

printf "\n"

rm "$GITHUB_TMP_FILE" 2>/dev/null

echo "Done!"
