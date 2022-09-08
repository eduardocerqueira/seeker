#date: 2022-09-08T17:16:28Z
#url: https://api.github.com/gists/448f8d6d7bf907c9e9976b4bf2069fb1
#owner: https://api.github.com/users/Zeitsperre

import logging
from datetime import datetime as dt
from getpass import getpass
from pathlib import Path

from paramiko import SSHClient
from scp import SCPClient

logging.basicConfig(
    filename=f"{ dt.strftime(dt.now(), '%Y-%m-%d')}_{Path(__file__).stem}.log",
    level=logging.INFO,
)  # can't go wrong making a logfile

user = "my_user_name"  # username on the server being accessed.
my_folder = Path("/to/my/folder/")  # folder being transferred

with SSHClient() as ssh:
    ssh.load_system_host_keys()  # loads any SSH keys. If keys are loaded, password not needed.

    server_address = f"server.host.ca"
    logging.info(f"Connecting to {server_address}")
    pw = getpass("Password: "**********"
    
    ssh.connect(server_address, username= "**********"=pw)  # opens a shell to create a connection.
    logging.info(f"Connected!")

    with SCPClient(ssh.get_transport(), socket_timeout=30.0) as scp:
         scp.put(my_folder, recursive=True, remote_path='/home/user/whatever')
er/whatever')
