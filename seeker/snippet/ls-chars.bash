#date: 2023-07-28T16:37:29Z
#url: https://api.github.com/gists/e3e01aa9959e2640d8c63344bbfe6d47
#owner: https://api.github.com/users/airvzxf

#!/usr/bin/env bash

# Thanks for this code Lu Xu.
# https://stackoverflow.com/a/60475015/1727383

# Make this file executable.
# chmod u+x ls-chars.bash

# It needs the package: 'fontconfig'.

# Check the list of installed fonts.
# fc-match --all | grep --color=always -ins awesome

count=0

for range in $(fc-match --format='%{charset}\n' "${1}"); do
  for n in $(seq "0x${range%-*}" "0x${range#*-}"); do
    printf "%05x\n" "${n}"
  done
done | while read -r n_hex; do
  count=$((count + 1))
  printf "%-6s\U${n_hex} | " "${n_hex}"
  [ $((count % 10)) = 0 ] && printf "\n"
done
printf "\n"
