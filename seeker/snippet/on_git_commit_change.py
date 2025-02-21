#date: 2025-02-21T16:46:59Z
#url: https://api.github.com/gists/6f2c8f2574db6df770c51795d02cd458
#owner: https://api.github.com/users/gwpl

#!/usr/bin/env python3 

# TL;DR- Put it in PATH and `chmod +x` and enjoy `on_git_commit_change.py -c -- command to run every time I make new commit`

# https://gist.github.com/gwpl/6f2c8f2574db6df770c51795d02cd458

import argparse
import subprocess
import time
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Monitor Git commit changes and execute a command.")
    parser.add_argument('-f', '--frequency', type=float, default=0.5, help='Frequency of checking for new commits in seconds.')      
    parser.add_argument('-s', '--silent', action='store_true', help='Run in silent mode.')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='Command to execute on new commit.')
    parser.add_argument('-a', '--after-change', action='store_true', help='Execute command only after observing a change.')
    parser.add_argument('-c', '--clear', action='store_true', help='Clear the terminal before logging new commit.')
    return parser.parse_args()

def log_info(message, silent):
    if not silent:
        print(f"INFO: {message}", file=sys.stderr)

def get_current_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')

def main():
    args = parse_arguments()
    if not args.command:
        print("No command provided to execute on commit change.", file=sys.stderr)
        sys.exit(1)

    current_commit = None
    if args.after_change:
        # If the after-change flag is set, initialize current_commit to the current HEAD
        current_commit = get_current_commit()
        log_info(f"Current commit: {current_commit}", args.silent)

    while True:
        new_commit = get_current_commit()
        if new_commit != current_commit:
            if args.clear:
                subprocess.run(['clear'])
            log_info(f"New commit detected: {new_commit}", args.silent)
            current_commit = new_commit
            subprocess.run(args.command[1:])
        time.sleep(args.frequency)

if __name__ == "__main__":
    main()
