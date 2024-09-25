#date: 2024-09-25T16:54:34Z
#url: https://api.github.com/gists/448c831143c89fa986703551428253e5
#owner: https://api.github.com/users/esimkowitz

declare -a issues=($( gh search issues "created:<2024-09-22" "is:open" --repo "wavetermdev/waveterm" --json url --jq ".[] | .url" --limit 200 ))

for url in "${issues[@]}"
do
	gh issue edit $url --add-label "legacy"
done