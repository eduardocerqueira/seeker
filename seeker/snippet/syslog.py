#date: 2024-04-12T17:03:15Z
#url: https://api.github.com/gists/338fe574dff93ed64c30cee28aef3670
#owner: https://api.github.com/users/Sovenok-Hacker

#!/usr/bin/env python

## Tiny Syslog Server in Python

HOST, PORT = "0.0.0.0", 514
GOTIFY_URL = "http://127.0.0.1:8050" # Your Gotify instance URL
GOTIFY_TOKEN = "**********"

import socketserver as ss
import requests

class SyslogUDPHandler(ss.BaseRequestHandler):
    def handle(self):
        data = self.request[0].strip().decode()
        addr = self.client_address[0]
        print(f"[SYSLOG {addr}] {data}")
        requests.post(
            GOTIFY_URL + "/message",
            params={
                "token": "**********"
            },
            json={
                "title": "Syslog message",
                "message": data
            }
        )

if __name__ == "__main__":
    try:
        server = ss.UDPServer((HOST, PORT), SyslogUDPHandler)
        server.serve_forever(poll_interval=0.2)
    except (IOError, SystemExit):
        raise
    except KeyboardInterrupt:
        print("Crtl+C Pressed. Shutting down.")("Crtl+C Pressed. Shutting down.")