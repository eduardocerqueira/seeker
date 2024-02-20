#date: 2024-02-20T16:58:23Z
#url: https://api.github.com/gists/fa9f00182d4c739a52e707f1a34972d5
#owner: https://api.github.com/users/justinc1

#!/usr/bin/env python3

import sys
from dataclasses import dataclass

_help_text="""VPN over SSH port forward helper. It generates opts for ssh binary.

Usage1: {sys.argv[0]} -L hostA port1 port2 -R hostB port3

Usage2 - ssh to bastionA to gain "direct" access to hostA ports 8080 and 8443
sudo ip addr add vmA/32 dev lo
ssh ubuntu@bastionA $(sys.argv[0]} -L hostA 8080 8443)
"""


@dataclass
class PFSpec:
    # port forward spec
    direction: str
    host: str
    ports: list[str]


def main():
    if len(sys.argv) < 4:
        sys.stderr.write(_help_text)
        sys.exit(1)

    argv = sys.argv[1:]
    direction_all = ["-L", "-R"]
    pfspecs = []
    ii_direction = [ii for ii, arg in enumerate(argv) if arg in direction_all]
    for jj, ii in enumerate(ii_direction):
        direction = argv[ii]
        assert direction in direction_all
        host = argv[ii+1]
        if jj+1 < len(ii_direction):
            ii_next_dir = ii_direction[jj+1]
        else:
            # last iteration
            ii_next_dir = len(argv)
        ports = argv[ii+2 : ii_next_dir]
        # print(f"DBG dir={direction} host={host} ports={ports}")
        pfspecs.append(PFSpec(direction, host, ports))

    outs = ""
    for pfs in pfspecs:
        for port in pfs.ports:
            outs += f"{pfs.direction} {pfs.host}:{port}:{pfs.host}:{port} "
    outs = outs.strip()
    print(outs)


if __name__ == "__main__":
    main()
