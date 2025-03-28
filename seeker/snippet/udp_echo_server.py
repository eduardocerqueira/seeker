//date: 2025-03-28T16:45:50Z
//url: https://api.github.com/gists/50e8f53a1d778e66186d9ac9b2af55da
//owner: https://api.github.com/users/winlinvip

import socket
import argparse

def udp_echo_server(host='0.0.0.0', port=10099):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((host, port))

    print(f"UDP Echo Server is listening on {host}:{port}...")

    while True:
        data, addr = server_socket.recvfrom(1024)
        print(f"Received from {addr}: {data}")
        server_socket.sendto(data, addr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UDP Echo Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to listen on')
    parser.add_argument('--port', type=int, default=20099, help='Port to listen on')

    args = parser.parse_args()
    udp_echo_server(args.host, args.port)
