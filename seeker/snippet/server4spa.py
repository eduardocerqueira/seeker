#date: 2024-07-01T16:35:42Z
#url: https://api.github.com/gists/51a9ff897514507334f635a14d6640e7
#owner: https://api.github.com/users/ec-c

#!/usr/bin/env python

# Inspired by https://gist.github.com/jtangelder/e445e9a7f5e31c220be6
# Python3 http.server for Single Page Application

import urllib.parse
import http.server
import socketserver
import re
from pathlib import Path

HOST = ('0.0.0.0', 3000)
pattern = re.compile('.png|.jpg|.jpeg|.js|.css|.ico|.gif|.svg|.wasm', re.IGNORECASE)


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        url_parts = urllib.parse.urlparse(self.path)
        request_file_path = Path(url_parts.path.strip("/"))

        ext = request_file_path.suffix
        if not request_file_path.is_file() and not pattern.match(ext):
            self.path = 'index.html'

        return http.server.SimpleHTTPRequestHandler.do_GET(self)


httpd = socketserver.TCPServer(HOST, Handler)
httpd.serve_forever()