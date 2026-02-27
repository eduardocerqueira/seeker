#date: 2026-02-27T17:19:03Z
#url: https://api.github.com/gists/ce7b5eba4aeb00213e1e85461db67be0
#owner: https://api.github.com/users/andrewhaller

#!/bin/sh

# Display all character classes defined in the current locale

# This script uses the `locale` command to list all character classes available
# in the LC_CTYPE category of the current locale settings. For each character
# class, it uses `recode` to filter characters belonging to that class and
# displays them in a human-readable format using `od`.

if type recode >/dev/null 2>&1 && type locale >/dev/null 2>&1; then
    for __charclass in $(
        locale -v LC_CTYPE |
            sed 's/combin.*//;s/;/\n/g;q'
    ); do
        printf "\n\t%s\n\n" "$__charclass"
        recode u2/test16 -q </dev/null |
            tr -dc "[:$__charclass:]" |
            od -A n -t a -t o1z -w12
    done && unset __charclass
fi