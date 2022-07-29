#date: 2022-07-29T16:52:18Z
#url: https://api.github.com/gists/43098fa164657e9f5d2acaeaa4da2003
#owner: https://api.github.com/users/Methacrylon

import asyncio
from asyncssh.public_key import import_private_key

from asyncssh.connection import SSHClientConnection, connect


SSH_PRIVKEY = """-----BEGIN RSA PRIVATE KEY-----
**************************************
-----END RSA PRIVATE KEY-----"""


async def ls(ssh_conn: SSHClientConnection, directory: str):
    return await ssh_conn.run(f"ls {directory}")


async def main() -> None:
    ssh_username = "tamles"
    ssh_host = "localhost"
    async with connect(
        ssh_host, username=ssh_username, client_keys=[import_private_key(SSH_PRIVKEY)]
    ) as ssh_conn:
        files = await ls(ssh_conn, ".")
    print(files.stdout)


if __name__ == "__main__":
    asyncio.run(main())