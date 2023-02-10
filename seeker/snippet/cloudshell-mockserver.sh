#date: 2023-02-10T16:45:29Z
#url: https://api.github.com/gists/6d0e4fdd9f5d25e794e837650100f063
#owner: https://api.github.com/users/micovery

#!/bin/bash

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

function get_free_port() {
  read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
  while :; do shuffPort="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"; netstat -tapln 2> /dev/null | grep $shuffPort || break; done
  echo "${shuffPort}"
}

function run_mock_server(){
    
    port=$(get_free_port)
    message=${1:-"Hello World"}
    
    python3 -c """
import http.server
import socketserver
from http import HTTPStatus


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()
        self.wfile.write(b'""${message}""')


httpd = socketserver.TCPServer(('', """${port}"""), Handler)
httpd.serve_forever()
    """ &> /dev/null &

    ssh -oStrictHostKeyChecking=no -q -R  80:localhost:${port} ssh.localhost.run | head -1 

}

run_mock_server "$@"