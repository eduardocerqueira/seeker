#date: 2025-07-09T16:49:48Z
#url: https://api.github.com/gists/e638db412200476fdc8ff418f17ecd74
#owner: https://api.github.com/users/levihuayuzhang

cd "/Library/Application Support/com.apple.idleassetsd/Customer" && cat entries.json | jq -r '.assets[] | (.id + "," + .["url-4K-SDR-240FPS"])' | parallel wget --no-check-certificate -q -O './4KSDR240FPS/{= s:\,[^,]+$::; =}.mov' '{= s:[^,]+\,::; =}';