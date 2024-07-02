#date: 2024-07-02T16:49:34Z
#url: https://api.github.com/gists/27367cadfff22a99a017be37f95c1bc7
#owner: https://api.github.com/users/lcatlett

#!/usr/bin/env bash
# Runs a command as a background job gets exit code of child processes.
# Allows for more granular error handling and logging of Pantheon / terminus commands.

set -m      # allow for job control
EXIT_CODE=0 # exit code of overall script

# may need to set -o posix if we run into issues with the for loop returning a 127 or bad_trap error
INPUT_CMD=$1

function wait_and_get_exit_codes() {
    children=("$@")
    EXIT_CODE=0
    for job in "${children[@]}"; do
        echo "PID => ${job}"
        CODE=0
        wait ${job} || CODE=$?
        if [[ "${CODE}" != "0" ]]; then
            echo "At least one test failed with exit code => ${CODE}"
            EXIT_CODE=1
        fi
    done
}

DIRN=$(dirname "$0")

commands=(
    "{ $INPUT_CMD; }"
)

clen=$(expr "${#commands[@]}" - 1) # get length of commands - 1

children_pids=()
for i in $(seq 0 "$clen"); do
    (echo "${commands[$i]}" | bash) & # run the command via bash in subshell
    children_pids+=("$!")
    echo "$i ith command has been issued as a background job"
done

wait_and_get_exit_codes "${children_pids[@]}"

echo "EXIT_CODE => $EXIT_CODE"
exit "$EXIT_CODE"
