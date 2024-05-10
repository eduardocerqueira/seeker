#date: 2024-05-10T17:05:40Z
#url: https://api.github.com/gists/986b033d315ee1f5e25ed6c7d2b50820
#owner: https://api.github.com/users/agatemosu

import asyncio
import subprocess

HOST = "0.0.0.0"
PORT = 62775


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info("peername")
    client_ip = f"{addr[0]}:{addr[1]}"
    print(f"Connection with {client_ip} opened.")

    while True:
        data = await reader.read(1024)

        if not data:
            break

        if data == b"reboot":
            subprocess.run(["shutdown", "/r", "/t 0"])

    print(f"Connection with {client_ip} closed.")
    writer.close()


async def main():
    server = await asyncio.start_server(handle_client, HOST, PORT)

    for socket in server.sockets:
        addr = socket.getsockname()
        server_ip = f"{addr[0]}:{addr[1]}"
        print(f"Serving on {server_ip}.")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server closed.")
