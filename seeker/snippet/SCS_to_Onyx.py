#date: 2022-03-04T16:53:02Z
#url: https://api.github.com/gists/a51b2a4dee9063ae8a96248f29499159
#owner: https://api.github.com/users/sleepsleeprepeat

from telnetlib import Telnet
from http.server import HTTPServer, BaseHTTPRequestHandler

WEBSERVER_PORT = 8000
TELNET_PORT = 2323

TELCON = Telnet("localhost", 2323)


def sendTelnet():
    TELCON.write("GQL 1".encode("ascii") + b"\n")


class handleRequest(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("content-type", "text/html")
        self.end_headers()
        self.wfile.write("OK".encode())
        sendTelnet()


def main():
    server = HTTPServer(("", WEBSERVER_PORT), handleRequest)
    server.serve_forever()


if __name__ == "__main__":
    main()
