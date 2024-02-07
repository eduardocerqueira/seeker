#date: 2024-02-07T17:01:12Z
#url: https://api.github.com/gists/cdbb6269e74f94ace30b04bb47d7c5cf
#owner: https://api.github.com/users/tgran2028

#!/usr/bin/env python3

"""Open Oracle Solaris man page URL"""

import re
import sys
import webbrowser
from pathlib import Path

import plumbum as pb


def get_manpage_name(name: str) -> str:
    """Get name of manpage"""
    man = pb.local["man"]
    try:
        p = man["-w"](name).strip("\n")
    except pb.ProcessExecutionError as e:
        sys.stderr.write(f'manpage not found for "{name}": {e}')
        raise e

    return Path(p).name.rstrip(".gz")


def convert_manpage_name_for_uri(name) -> str:
    """Name adjustments to match oracle URI pattern"""
    if m := re.search(r".\d$", name):
        return name.replace(m.group(), m.group().replace(".", "-"))
    return name


def open_oracle_manpage(name) -> None:
    manpage_name = get_manpage_name(sys.argv[1])
    url = f"https://docs.oracle.com/cd/E88353_01/html/E37839/{convert_manpage_name_for_uri(manpage_name)}.html"
    webbrowser.open(url, new=2)


if __name__ == "__main__":
    name = sys.argv[1]
    open_oracle_manpage(name)
