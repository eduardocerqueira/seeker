#date: 2024-06-27T16:59:34Z
#url: https://api.github.com/gists/b8c652f176afc50475815758a648068b
#owner: https://api.github.com/users/AlecsFerra

#! /usr/bin/env python3

# flatten.py
# Copyright (C) 2024 Alessio Ferrarini <github.com/alecsferra>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>


import os
import re
import argparse

def flatten(root_dir, file_name):
    content = ""
    try:
        full_path = os.path.join(root_dir, file_name)
        with open(full_path, "r") as f:
            content = f.readlines()
    except:
        print(f"File '{full_path}' not found") # type: ignore

    def do_flatten(line):
        matched = re.match(r"\\(sub)?import{(.*)}{(.*)}", line)
        if matched:
            dir = matched.group(2)
            file = f"{matched.group(3)}.tex"
            return flatten(os.path.join(root_dir, dir), file)
        
        return line

    return ''.join(map(do_flatten, content)) # type: ignore

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="flatten.py",
        description="Tranform a latex project that uses the '\\(sub)import' in"
            " a standalone latex file"
    )

    parser.add_argument(
        "source", 
        help="The main file of the latex project",
    )

    parser.add_argument(
        "output", 
        help="The name of the file used as output"
    )

    args = parser.parse_args()

    out = flatten(".", args.source)

    with open(args.output, "w") as f:
        f.write(out)
