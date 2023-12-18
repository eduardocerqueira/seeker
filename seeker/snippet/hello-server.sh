#date: 2023-12-18T16:59:16Z
#url: https://api.github.com/gists/bdbbd1c69a2afc76ab9a47114c858c44
#owner: https://api.github.com/users/nfiles

#! /bin/bash

set -e
set -o pipefail

usage() {
	echo "$0 [--port <PORT>]"
}

PORT=8888

while ARG="$1"; shift; do
	case "$ARG" in
		"--port" | "-p") PORT="$1"; shift ;;
		"--help" | "-h") usage; exit 0 ;;
		*) usage; exit 1 ;;
	esac
done

while true; do
	printf "HTTP/1.1 200 OK\r\nHost: 127.0.0.1:$PORT\r\nServer: netcat!\r\nContent-Type: application/json; charset-UTF-8\r\nContent-Length: 34\r\n\r\n{ \"message\": \"Hello, world!\" }\r\n\r\n" \
		| netcat -l "$PORT"
done
