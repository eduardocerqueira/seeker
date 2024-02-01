#date: 2024-02-01T16:47:39Z
#url: https://api.github.com/gists/257ec5bc93a7ba429eb56bd4022515c5
#owner: https://api.github.com/users/nparkhe83

history | tail -r | awk '!seen[$2]++ { print $0 "\n" }' | tail -r