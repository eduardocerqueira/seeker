#date: 2024-05-10T17:05:40Z
#url: https://api.github.com/gists/986b033d315ee1f5e25ed6c7d2b50820
#owner: https://api.github.com/users/agatemosu

import asyncio

HOST = "127.0.0.1"
PORT = 62775


async def send_reboot():
    try:
        _, writer = await asyncio.open_connection(HOST, PORT)

        writer.write(b"reboot")
        await writer.drain()

        writer.close()
        await writer.wait_closed()

    except ConnectionError as error:
        print(error.strerror)


async def main():
    print("Sending reboot to", f"{HOST}:{PORT}")
    await send_reboot()


if __name__ == "__main__":
    asyncio.run(main())
