#date: 2025-04-24T16:54:39Z
#url: https://api.github.com/gists/37b0780fab553d0522b24748f546b955
#owner: https://api.github.com/users/ig-gh

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import requests
import re

AZURE_API_ENDPOINT = "<YOUR_DEPLOYMENT_ENDPOINT_URL>"
                   # "https://****************.cognitiveservices.azure.com/openai/deployments/o3-mini"
AZURE_API_VERSION = "<DEPLOYMENT_API_VERSION>"
                  # "2025-01-01-preview"
AZURE_API_KEY = "<YOUR_API_KEY>"
PROXY_PORT = 8000

class ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
                # 1. Read the request from Zed
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        try:
            zed_payload = json.loads(post_data.decode('utf-8'))
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"Invalid JSON from Zed")
            return

        # 2. Transform the request for Azure OpenAI
        azure_payload = self.transform_request(zed_payload)

        # 3. Forward the request to Azure OpenAI (streaming)
        azure_url = f"{AZURE_API_ENDPOINT}/chat/completions?api-version={AZURE_API_VERSION}"

        try:
            response = requests.post(
                azure_url,
                headers={
                    "Content-Type": "application/json",
                    "api-key": AZURE_API_KEY,
                    "Accept": "text/event-stream",  # Important: request SSE
                    "user-agent": "Zed/0.178.5"
                },
                json=azure_payload,
                stream=True  # Enable streaming
            )
            response.raise_for_status()

            # Set headers for SSE streaming to Zed
            self.send_response(200)
            self.send_header('Content-type', 'text/event-stream')  #Important for SSE.
            self.send_header('Cache-Control', 'no-cache')  # Disable caching
            self.end_headers()

            # Stream and filter the Azure OpenAI response to Zed
            try:
                # Use iter_content for potentially more accurate SSE proxying
                for chunk in response.iter_content(chunk_size=None):
                    if not chunk:
                        continue # Skip empty chunks if any

                    # --- Filtering Logic ---
                    line = chunk.decode('utf-8').strip()

                    # 1. Pass through the DONE signal directly
                    if line == 'data: [DONE]':
                        # print(f"Forwarding DONE signal: {chunk}")
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        continue # Move to next chunk

                    # 2. Check if it's a data event
                    if line.startswith('data: '):
                        json_str = line[len('data: '):].strip()
                        if not json_str: # Handle cases like 'data: \n'
                           continue

                        try:
                            # 3. Parse the JSON payload
                            data_payload = json.loads(json_str)

                            # 4. Check for the 'choices' key and if it's empty
                            if 'choices' in data_payload and data_payload['choices'] == []:
                                continue # Skip this chunk

                            # 5. If choices are not empty or key doesn't exist (shouldn't happen for content), forward it
                            self.wfile.write(chunk)
                            self.wfile.flush()

                        except json.JSONDecodeError:
                            # Should not happen with valid Azure responses, but good practice
                            # print("Warning: Could not decode JSON in chunk, forwarding anyway.")
                            self.wfile.write(chunk) # Forward potentially non-JSON data lines if needed
                            self.wfile.flush()
                    else:
                         # Forward lines that don't start with 'data: ' (e.g., comments, empty lines between events)
                         # Although Azure usually only sends 'data:' lines and the final DONE.
                         self.wfile.write(chunk)
                         self.wfile.flush()
                    # --- End Filtering Logic ---

            except BrokenPipeError:
                print("Client disconnected during streaming (expected if Zed errored or user cancelled).")
            except Exception as e:
                print(f"An error occurred during streaming to client: {e}")
            # finally:
                # response.close() # Usually handled by requests context manager or exiting loop

        except requests.exceptions.RequestException as e:
            # Check if headers were already sent before trying to send an error response
            # (Need to track this state or check a flag if BaseHTTPRequestHandler doesn't provide one easily)
            # Simplified: Assume if we get here, headers weren't sent. If they were, this will error too.
            try:
                 self.send_response(502)
                 self.send_header('Content-type', 'text/plain')
                 self.end_headers()
                 self.wfile.write(f"Error communicating with Azure OpenAI: {e}".encode('utf-8'))
            except Exception as send_error:
                print(f"Error communicating with Azure OpenAI: {e}. Could not send error response to client: {send_error}")
                return
            try:
                 self.send_response(500)
                 self.send_header('Content-type', 'text/plain')
                 self.end_headers()
                 self.wfile.write(b"Internal Server Error in Proxy")
            except Exception as send_error:
                 print(f"Could not send 500 error response to client: {send_error}")



    def transform_request(self, zed_payload):
        # Define your system prompt.
        # ********************************************************************
        # *** THIS IS NECESSARY TO HAVE CODE HIGHLIGHTING IN THE RESPONSES ***
        # ********************************************************************
        system_message = {
            "role": "system",  
            "content": "You must always respond in markdown format."
        }
        
        if 'prompt' in zed_payload:
            # Build the messages array so that system prompt comes first.
            messages = [
                system_message,
                {"role": "user", "content": zed_payload['prompt']}
            ]
            azure_payload = {
                "messages": messages,
                "temperature": zed_payload.get("temperature", 0.7),
                "max_tokens": "**********"
                "stream": True
            }
            zed_payload.pop("prompt")
            # If there are extra parameters in zed_payload, update them now.
            azure_payload.update(zed_payload)
        else:
            azure_payload = zed_payload
            azure_payload["stream"] = True
            # Optionally add system message if not included already.
            if "messages" in azure_payload:
                azure_payload["messages"] = [system_message] + azure_payload["messages"]
            else:
                azure_payload["messages"] = [system_message]
        return azure_payload


if __name__ == '__main__':
    server_address = ('', PROXY_PORT)
    httpd = HTTPServer(server_address, ProxyHandler)
    print(f"Starting Azure OpenAI proxy server on port {PROXY_PORT}")
    httpd.serve_forever()
 httpd.serve_forever()
