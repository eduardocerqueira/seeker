#date: 2024-03-18T16:57:54Z
#url: https://api.github.com/gists/f717928297f2f00398c9d1c4467d2973
#owner: https://api.github.com/users/jakebrinkmann

#!/usr/bin/env python3
"""Remote Sudo (python, paramiko, ssh, ssh_config)"""
import os
import sys
from glob import glob

import paramiko


def remote_sudo(cmd, host, user, passwd, port=22):
    ssh=paramiko.SSHClient()
    ssh._policy=paramiko.WarningPolicy()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh_config=paramiko.SSHConfig()
    user_cfg_file=glob(os.path.expanduser("~/.ssh/config"))
    if user_cfg_file:
        ssh_config.parse(open(user_cfg_file[0]))

    cfg={
        "hostname": host,
        "username": user,
        "password": "**********"
        "port": port,
        "timeout": 60,
    }

    user_config=ssh_config.lookup(cfg["hostname"])
    for k in("hostname", "username", "port"):
        if k in user_config:
            cfg[k] =user_config[k]

    print(f"Connect to {cfg['hostname']}as {cfg['username']}...")
    ssh.connect(**cfg)
    transport=ssh.get_transport()
    session=transport.open_session()
    session.set_combine_stderr(True)
    session.get_pty()

    # for testing purposes we want to force sudo to always to ask for password. because of that we use "-k" key
    session.exec_command(f"sudo -k {cmd}")
    stdin=session.makefile("wb", -1)
    stdout=session.makefile("rb", -1)
    # you have to check if you really need to send password here
    stdin.write(passwd+"\n")
    stdin.flush()

    returnstdout.read().splitlines()


def main(host="10.192.1.1", user="lookatmenow", passwd="superS3cre7"):
    cmd = "echo $PS1"
    return remote_sudo(cmd, host, user, passwd)


if __name__=="__main__":
    host=sys.argv[1]
    print(main(host))
