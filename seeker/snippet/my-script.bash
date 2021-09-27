#date: 2021-09-27T16:58:41Z
#url: https://api.github.com/gists/1d694d66cffa3338920b63520c65bfa4
#owner: https://api.github.com/users/tychobrailleur

#!/usr/bin/env bash

# Call script with:
#    VAL=pouet ./my-script.bash

var_val="${VAL}"
echo ${var_val}

output="$(
       echo "[${VAL}]"
)"
echo $output

output="$(
       VAL=pouetpouet; echo "[${VAL}]"
)"
echo $output


output="$(
       VAL=pouetpouetpouet echo "[${VAL}]"
)"
echo $output
