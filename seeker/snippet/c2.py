#date: 2022-02-25T17:01:47Z
#url: https://api.github.com/gists/be6f0a793a7b7568b71214c35a742d96
#owner: https://api.github.com/users/hattmo

import socket
import sys
from typing import List, TypedDict

class Settings(TypedDict):
    port:int


def process_args(args:List[str])->Settings:
    try:
        port = int(args[1])
    except:
        port = 1337
    return Settings(port=port)


def main():
    settings = process_args(sys.argv)
    print(f"Configuration: {{{settings}}}")
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0",settings['port']))
        s.listen()
        while True:
            print("Waiting for callback")
            conn, addr = s.accept()
            print("Client connected")
            data = conn.recv(1024)
            print(f"Handshake: {{{data}}}")
            if data != b"HELLO\n":
                print("Invalid handshake")
                conn.close()
                continue
            print(f"Valid connection from {addr[0]}")
            command = input("Command: ")
            command_bytes = command.encode() + b"\0"
            conn.sendall(command_bytes)
            print("Waiting for response")
            buf = b''
            while True:
                try:
                    buf += conn.recv(1024)
                    if not buf:
                        break
                except ConnectionResetError:
                    break
            print(f"Response: {{{buf}}}")
            conn.close()

if __name__ == '__main__':
    main()
