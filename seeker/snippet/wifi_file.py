#date: 2025-02-18T17:12:36Z
#url: https://api.github.com/gists/e0ed3d8435343dd6edfb947f358736c4
#owner: https://api.github.com/users/xzripper

from http.server import SimpleHTTPRequestHandler

from socketserver import TCPServer

from socket import socket, AF_INET, SOCK_DGRAM


CONST_PORT = 1111
CONST_FILE = 'REPLACE-ME' # Name of the file in the current directory.

def get_local_ip():
    s_inst = socket(AF_INET, SOCK_DGRAM)

    try:
        s_inst.connect(("8.8.8.8", 80))

        ip = s_inst.getsockname()[0]
    finally:
        s_inst.close()

    return ip

local_ip = get_local_ip()

class DownloadHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(302)

            self.send_header('Location', f'/{CONST_FILE}')
            self.end_headers()

        elif self.path == f'/{CONST_FILE}':
            self.send_response(200)

            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Disposition', f'attachment; filename="{CONST_FILE}"')

            self.end_headers()

            with open(CONST_FILE, 'rb') as file:
                self.wfile.write(file.read())
        else:
            super().do_GET()

with TCPServer(("", CONST_PORT), DownloadHandler) as httpd:
    print(f"Serving at http://{local_ip}:{CONST_PORT}")

    httpd.serve_forever()
