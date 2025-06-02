#date: 2025-06-02T17:02:43Z
#url: https://api.github.com/gists/c2293292b6f105dd9e4ef25811b54b55
#owner: https://api.github.com/users/alexmohr

import socket
import re

def get_content_length(request):
    match = re.search(r'Content-Length: (\d+)', request)
    if match:
        return int(match.group(1))
    return None

def start_server(port):
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Bind the socket to all interfaces on the given port
    server_socket.bind(('0.0.0.0', port))
    
    # Start listening for incoming connections
    server_socket.listen(5)
    print(f"Server listening on port {port}...")
    
    while True:
        # Accept a connection
        client_socket, addr = server_socket.accept()
        print(f"\nConnection from {addr}")
        
        # Receive the initial request to get headers
        request = b''
        while True:
            chunk = client_socket.recv(1024)
            request += chunk
            if b'\r\n\r\n' in request:
                break
        
        headers = request.decode('utf-8')
        content_length = get_content_length(headers)
        
        if content_length:
            remaining_bytes = content_length - (len(request) - headers.index('\r\n\r\n') - 4)
            while remaining_bytes > 0:
                chunk = client_socket.recv(min(1024, remaining_bytes))
                request += chunk
                remaining_bytes -= len(chunk)
        
        print(request.decode('utf-8'))
        
        # Close the connection
        client_socket.close()

if __name__ == "__main__":
    port = 8080  # You can change the port number as needed
    start_server(port)