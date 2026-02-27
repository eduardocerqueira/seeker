#date: 2026-02-27T17:14:16Z
#url: https://api.github.com/gists/de3edcc9a1423c1ae3a19d3606e388f7
#owner: https://api.github.com/users/andrewhaller

#!/bin/sh

# Display all printable characters

# Loop from decimal 32 to 126 (space to tilde) and print the corresponding ASCII characters
for __index in $(seq 32 126); do
    # Convert decimal to octal format (e.g., 32 becomes \040, 126 becomes \176)
    __OCTAL=$(printf '\\%o' "$__index")
    # Use printf with %b to interpret the octal escape and print the character
    printf "%b" "$__OCTAL"
done && unset __index __OCTAL
