#date: 2023-10-11T17:09:04Z
#url: https://api.github.com/gists/e111ef66a0c19f6cfaa078db2f3033ea
#owner: https://api.github.com/users/quantumcore

import paramiko
from scp import SCPClient
import argparse

def main():
    parser = argparse.ArgumentParser(description="Securely transfer files over SSH using SCP")
    parser.add_argument("ip", help="Remote host IP address")
    parser.add_argument("lfile", help="Local file path")
    parser.add_argument("rfile", help="Remote file path")
    parser.add_argument("ppkey", help="Path to private key file")
    parser.add_argument("keypass", help= "**********"
    parser.add_argument("username", help="SSH username")

    args = parser.parse_args()

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        key = "**********"=args.keypass)
        ssh.connect(args.ip, pkey=key, username=args.username)

        with SCPClient(ssh.get_transport()) as scp:
            print(f"uploading {args.lfile} to {args.ip} -> {args.rfile} ... ")
            scp.put(args.lfile, args.rfile)

    except Exception as e:
        print("Error: " + str(e))
        print("USAGE: ./script.py host local_file remote_file private_key private_keypass username")

if __name__ == "__main__":
    main()
ivate_keypass username")

if __name__ == "__main__":
    main()
