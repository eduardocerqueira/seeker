#date: 2024-03-19T16:47:15Z
#url: https://api.github.com/gists/f552cc0296438e8362cc70f9ab570cfc
#owner: https://api.github.com/users/Minemobs

#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Illegal number of parameters" >&2
    exit 2
fi

echo "Getting the Video provider's URL"
VIDEO_PROVIDER_URL=$(curl_ff109 -s $1 | grep -E "\s+video\[0\] = 'https://.+'" | cut -d "'" -f 2)
URL=""

fv_parse () {
    #Actual code
    echo "Getting the M3U8 file"
    local FV_GET_JS_SCRIPT=$(curl_ff109 -s $VIDEO_PROVIDER_URL | grep "https://fusevideo.io/f/u/u/u/u" | cut -d '"' -f 2)
    local FV_GET_M3U8_URL=$(curl_ff109 -s $FV_GET_JS_SCRIPT | grep -o -E "\(n=atob\(\".+=\")" | cut -d "\"" -f 2 | base64 --decode | grep -o -E "https:.+\"" | cut -d "\"" -f 1 | sed 's/\\\//\//g')
    echo "Parsing the M3U8"
    URL=$(curl_ff109 -s $FV_GET_M3U8_URL | grep "https://" | head -n 1)
}

case "$VIDEO_PROVIDER_URL" in
    "https://fusevideo.io"*)
        # echo "Fuse video"
        fv_parse
        echo "Done: " "$URL";;
    *)
        echo "$VIDEO_PROVIDER_URL" "isn't supported yet"
        exit 0;;
esac