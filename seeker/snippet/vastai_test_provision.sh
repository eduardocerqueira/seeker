#date: 2025-07-22T16:54:36Z
#url: https://api.github.com/gists/decd21b780b552da605a55d58d0e41a2
#owner: https://api.github.com/users/RKelln

#!/bin/bash

# Simple test provisioning script to verify PROVISIONING_SCRIPT execution
# This script creates multiple marker files to confirm it ran successfully

echo "🚀 TEST PROVISIONING SCRIPT STARTED at $(date)" | tee -a /tmp/test_provisioning.log

# Create marker files in multiple locations to be absolutely sure we can find them
echo "TEST PROVISIONING SUCCESS - $(date)" > /workspace/TEST_PROVISIONING_SUCCESS.txt
echo "TEST PROVISIONING SUCCESS - $(date)" > /tmp/TEST_PROVISIONING_SUCCESS.txt  
echo "TEST PROVISIONING SUCCESS - $(date)" > /root/TEST_PROVISIONING_SUCCESS.txt

# Log some environment info
echo "Environment check:" >> /tmp/test_provisioning.log
echo "PROVISIONING_SCRIPT: $PROVISIONING_SCRIPT" >> /tmp/test_provisioning.log
echo "GITHUB_ACCESS_TOKEN: "**********":0:10}..." >> /tmp/test_provisioning.log
echo "PWD: $PWD" >> /tmp/test_provisioning.log
echo "USER: $USER" >> /tmp/test_provisioning.log
echo "HOME: $HOME" >> /tmp/test_provisioning.log

# Create a simple verification endpoint on port 9999
cat > /tmp/test_verification.py << 'EOF'
#!/usr/bin/env python3
import http.server
import socketserver
import os
import datetime

PORT = 8000

class TestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/test':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            response = """
            <html><head><title>Provisioning Script Test</title></head><body>
            <h1>🎉 PROVISIONING SCRIPT EXECUTED SUCCESSFULLY!</h1>
            <p><strong>This confirms that the PROVISIONING_SCRIPT environment variable is working correctly.</strong></p>
            <p>Test executed at: {timestamp}</p>
            <hr>
            <h2>Marker Files Status:</h2>
            <ul>
            """.format(timestamp=datetime.datetime.now())
            
            # Check for marker files
            marker_locations = [
                '/workspace/TEST_PROVISIONING_SUCCESS.txt',
                '/tmp/TEST_PROVISIONING_SUCCESS.txt', 
                '/root/TEST_PROVISIONING_SUCCESS.txt'
            ]
            
            for location in marker_locations:
                if os.path.exists(location):
                    try:
                        with open(location, 'r') as f:
                            content = f.read().strip()
                        response += f"<li>✅ <strong>{location}</strong>: {content}</li>"
                    except Exception as e:
                        response += f"<li>❓ <strong>{location}</strong>: exists but couldn't read ({e})</li>"
                else:
                    response += f"<li>❌ <strong>{location}</strong>: not found</li>"
            
            # Show log contents if available
            if os.path.exists('/tmp/test_provisioning.log'):
                try:
                    with open('/tmp/test_provisioning.log', 'r') as f:
                        log_content = f.read()
                    response += f"""
                    </ul>
                    <hr>
                    <h2>Provisioning Log:</h2>
                    <pre style="background:#f0f0f0; padding:10px; border:1px solid #ccc;">{log_content}</pre>
                    """
                except:
                    response += "</ul><p>Log file exists but couldn't read it.</p>"
            else:
                response += "</ul><p>No log file found.</p>"
            
            response += "</body></html>"
            
            self.wfile.write(response.encode('utf-8'))
        else:
            # Default response for any other path
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Test provisioning script verification server is running. Visit /test endpoint.')

# Start the server
try:
    with socketserver.TCPServer(("", PORT), TestHandler) as httpd:
        print(f"Test verification server started on port {PORT}")
        httpd.serve_forever()
except Exception as e:
    print(f"Failed to start test server: {e}")
EOF

# Start the test verification server in the background  
nohup python3 /tmp/test_verification.py > /tmp/test_server.log 2>&1 &

# Give it a moment to start
sleep 2

echo "✅ TEST PROVISIONING SCRIPT COMPLETED at $(date)" | tee -a /tmp/test_provisioning.log
echo "🌐 Test verification server started on port 9999 - visit /test endpoint to verify success" | tee -a /tmp/test_provisioning.log
echo "📁 Marker files created in /workspace, /tmp, and /root" | tee -a /tmp/test_provisioning.log

# Exit with success
exit 0
s
exit 0
