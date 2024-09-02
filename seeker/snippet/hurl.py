#date: 2024-09-02T17:05:25Z
#url: https://api.github.com/gists/75123b58d76b6618a40f83b2e47ce322
#owner: https://api.github.com/users/Phantop

from http.server import HTTPServer, BaseHTTPRequestHandler
from subprocess import run
from urllib.parse import urlparse, parse_qsl

class ServerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qsl(url.query)
        path = url.path

        command = f"hurl .{path}"  # Replace with your desired command
        for a,b in query:
            command += f" --variable {a}={b}"
        result = run(command, shell=True, capture_output=True)

        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(result.stdout)

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, ServerHandler)
    print(f"Server running on port {port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
