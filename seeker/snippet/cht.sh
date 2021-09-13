#date: 2021-09-13T17:01:27Z
#url: https://api.github.com/gists/0330e987db236e5d5fb69bc3a93b4714
#owner: https://api.github.com/users/iotku

#!/usr/bin/env bash
# cht.sh
# Original By ThePrimeagen, https://youtu.be/hJzqEAf2U4I
# Fixed/Modernized by iotku
languages=$(tr ' ' '\n' <<< "golang lua cpp c typescript nodejs")
core_utils=$(tr ' ' '\n' <<< "xargs find mv sed awk")

selected=$(printf "%s\n%s" "$languages" "$core_utils" | fzf)
read -rp "query: " query

if printf "%s" "$languages" | grep -qs "$selected"; then
    tmux neww bash -c "curl -s "cht.sh/$selected/$(tr ' ' '+' <<< "$query")" | less -R"
else
    tmux neww bash -c "curl -s "cht.sh/$selected~$query" | less -R"
fi
