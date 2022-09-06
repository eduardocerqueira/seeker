#date: 2022-09-06T17:17:01Z
#url: https://api.github.com/gists/5168d3fbeffab89b89d7b021f9711ac6
#owner: https://api.github.com/users/tsivinsky

#! /bin/bash

remote_addr="$1"
remote_port="$2"
local_port="$3"

pipe_file="netcat-tunnel-pipe"

function trap_ctrlc() {
	rm "$pipe_file"
	exit 2
}

trap "trap_ctrlc" 2

mkfifo "$pipe_file"

nc -k -l "$local_port" 0<$pipe_file | nc "$remote_addr" "$remote_port" 1>$pipe_file