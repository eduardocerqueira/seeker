#date: 2025-06-05T16:54:13Z
#url: https://api.github.com/gists/f8e42d7e110e2b40caca9f79a6ec1815
#owner: https://api.github.com/users/ab

#!/bin/bash

count="${1-1000}"
cmd="${2-/usr/bin/true}"

echo "Testing $count iterations of $cmd"

for (( i=0; i < count; i++ )); do
    "$cmd"
done
