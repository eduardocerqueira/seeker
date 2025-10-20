#date: 2025-10-20T17:07:40Z
#url: https://api.github.com/gists/05a1091ad362416b48e3a7d2c8dabd93
#owner: https://api.github.com/users/aozq

#!/bin/bash
# Simple working onstart script

echo "SIMPLE START: $(date)" > /tmp/simple_start.log

# Install pip3
echo "Installing pip3..." >> /tmp/simple_start.log
apt-get update -y >> /tmp/simple_start.log 2>&1
apt-get install -y python3-pip >> /tmp/simple_start.log 2>&1

# Install websockets
echo "Installing websockets..." >> /tmp/simple_start.log
pip3 install websockets >> /tmp/simple_start.log 2>&1

# Create simple HTTP server
echo "Creating simple HTTP server..." >> /tmp/simple_start.log
cat > /tmp/simple_server.py << 'EOF'
import http.server
import socketserver

class SimpleHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Simple Server Running")

PORT = 8888
with socketserver.TCPServer(("0.0.0.0", PORT), SimpleHandler) as httpd:
    print(f"Server running on 0.0.0.0:{PORT}")
    httpd.serve_forever()
EOF

# Start simple HTTP server
echo "Starting simple HTTP server..." >> /tmp/simple_start.log
python3 /tmp/simple_server.py > /tmp/server.log 2>&1 &

echo "SIMPLE START COMPLETE: $(date)" >> /tmp/simple_start.log
echo "Server PID: $!" >> /tmp/simple_start.log
