#date: 2022-06-15T17:13:06Z
#url: https://api.github.com/gists/affbda3f8c6b5c38648d4ab105777d88
#owner: https://api.github.com/users/azat

#!/usr/bin/env bash

function thread_worker()
{
    set -e

    # echo "thread_worker_pid=$BASHPID (started)"

    trap "STOP_THE_LOOP=1; sleep 0.5" INT
    STOP_THE_LOOP=0
    while [[ $STOP_THE_LOOP != 1 ]]; do
        sleep 0.1 | grep foo || :
    done

    # echo "thread_worker_pid=$BASHPID (finished)"
}

function main()
{
    echo "main_pid=$BASHPID"

    local pids=()
    for i in $(seq 0 100); do
        thread_worker &
        pids+=( $! )
    done

    sleep 0.1

    echo "Sending INT"
    wait &
    sleep 0.1
    kill -INT "${pids[@]}"
    sleep 0.1
    wait
}
main "$@"