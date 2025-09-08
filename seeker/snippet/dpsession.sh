#date: 2025-09-08T16:59:01Z
#url: https://api.github.com/gists/a0d0d160094f48c6ee1cadbe2ff1dc0d
#owner: https://api.github.com/users/e10withadot

if [[ -n "$1" ]]
then
    chosen_id="$1"
else
    list=$(devpod list | awk 'NR>3 {print $1}')
    if command -v tv >/dev/null 2>&1
    then
        chosen_id=$(echo "$list" | tv --no-preview)
    else
        if command -v fzf >/dev/null 2>&1
	then
            chosen_id=$(echo "$list" | fzf)
	else
	    echo "Usage: $0 <session-name>"
	    exit 1
	fi
    fi
fi
if [[ -n "$chosen_id" ]]
then
    devpod ssh "$chosen_id" --command "tmux -u new -A -s "$chosen_id""
fi
