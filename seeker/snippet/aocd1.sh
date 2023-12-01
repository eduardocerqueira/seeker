#date: 2023-12-01T16:54:14Z
#url: https://api.github.com/gists/cc0a4bccfde407bbf502397001ddbaae
#owner: https://api.github.com/users/wesgould

sed 's/[^0-9]//g' input | sed '/^[0-9]$/s/.*/&&/' | sed -n 's/\([0-9]\).*\([0-9]\)$/\1\2/p' | awk '{s+=$1} END {print s}'