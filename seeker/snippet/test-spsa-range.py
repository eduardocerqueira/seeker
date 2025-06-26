#date: 2025-06-26T17:03:09Z
#url: https://api.github.com/gists/ce741d2fc2e1be8d01396f35ceb3cbf6
#owner: https://api.github.com/users/MinetaS

import argparse
import subprocess
import sys

from typing import List

class SPSAParameter:
    def __init__(self, *args):
        self.name = args[0]
        self.initial_value = int(args[1])
        self.min_value = int(args[2])
        self.max_value = int(args[3])
        self.c_end = float(args[4])
        self.r_end = float(args[5])

class Context:
    def __init__(self, path: str):
        self.path = path
        self.p: subprocess.Popen
        self.params: List[SPSAParameter] = []

        if not self._read_parameters():
            self._abort(f"Failed to start Stockfish process from: {self.path}")

    def start(self):
        self._spawn_process()

        for param in self.params:
            self._test_parameter(param)

    def _abort(self, message: str | None = None):
        if self.p is not None:
            self.p.kill()
            if message is not None:
                sys.stderr.write(f"{message}\n")
            sys.exit(1)

    def _spawn_process(self):
        self.p = subprocess.Popen([self.path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, close_fds=True, universal_newlines=True)

    def _read_parameters(self):
        p = subprocess.Popen([self.path, "quit"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, close_fds=False, universal_newlines=True)
        info = p.stdout.readline().strip()

        if not (info.startswith("Stockfish") and info.endswith("(see AUTHORS file)")):
            return False

        for line in iter(p.stdout.readline, ""):
            tokens = "**********"

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"l "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********") "**********"  "**********"! "**********"= "**********"  "**********"6 "**********": "**********"
                continue

            self.params.append(SPSAParameter(*tokens))

        p.wait()
        print(f"Found {len(self.params)} tunable parameters")
        return True

    def _test_parameter_with_value(self, param: SPSAParameter, value: int):
        self.p.stdin.write(f"setoption name {param.name} value {value}\n")
        self.p.stdin.write("bench\n")

        for line in iter(self.p.stdout.readline, ""):
            line = line.rstrip("\r\n")

            if line.startswith("Nodes searched"):
                print(f"Parameter {param.name} = {value} passed (Nodes searched: {line.split(": ")[1]})")
                self.p.stdin.write(f"setoption {param.name} value {param.initial_value}\n")
                return

        print(f"Parameter {param.name} = {value} failed")

        self.p.wait()
        self._spawn_process()

    def _test_parameter(self, param: SPSAParameter):
        self._test_parameter_with_value(param, param.min_value)
        self._test_parameter_with_value(param, param.max_value)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to Stockfish executable file.")
    args = parser.parse_args()

    ctx = Context(args.path)
    ctx.start()


if __name__ == "__main__":
    main()
