#date: 2021-10-20T17:09:32Z
#url: https://api.github.com/gists/b19b6704112cf2eabcde8670475e0ccf
#owner: https://api.github.com/users/papajoker

#!/usr/bin/env python
"""
framework for make logs infos from yaml config file
"""

import sys
import os
from pathlib import Path
import subprocess
from datetime import datetime
import json
import locale
try:
    import yaml
except ValueError:
    print("Error: install python-yaml package : sudo pacman -S python-yaml")
    sys.exit(4)
import gi
try:
    girs = next(Path('/usr/share/gir-1.0/').glob('Pamac-*.gir')).stem[6:]
    gi.require_version('Pamac', girs) # import xml /usr/share/gir-1.0/Pamac-11.gir
    from gi.repository import Pamac
except StopIteration:
    print("Error: Pamac not installed")
    exit(4)


pamac_db = Pamac.Database(config=Pamac.Config(conf_path="/etc/pamac.conf"))

class Journald:
    def __init__(self, level = 4) -> None:
        self.logs = ()
        self.level = level

    def print_log(self, item, old_date):
        d = item['DATE'][:16]
        if d == old_date:
            d = ""
        print(f"{d:22}\n\t[{item['PRIORITY']}] {item['_UID']:4} {item['_CMDLINE']}\n\t{item['MESSAGE']}")
        return item['DATE'][:16]
        
    def print(self):
        max = 44
        i = 1
        old_date = ""
        for item in self.logs:
            i += 1
            old_date = self.print_log(item, old_date)
            if i > max:
                break
    
    def run(self):
        cmd = f'SYSTEMD_COLORS=0 /usr/bin/journalctl -b0 -p{self.level} --no-pager -o json'
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=-1, universal_newlines=True, text=True, shell=True) as proc:
            for line in proc.stdout:
                data = json.loads(line)
                item = {key: value for key, value in data.items() if key in ["PRIORITY", "MESSAGE", "_CMDLINE", "_UID", "__REALTIME_TIMESTAMP"]}
                dt_object = datetime.fromtimestamp(int(item["__REALTIME_TIMESTAMP"][0:10]))
                item['DATE'] = str(dt_object)
                if '_UID' not in item.keys():
                    item['_UID'] = "0"
                if '_CMDLINE' not in item.keys():
                    item['_CMDLINE'] = data["SYSLOG_IDENTIFIER"]
                self.logs = self.logs + (item,)
        if proc.returncode != 0:
            exit(proc.returncode)

def load_yaml(args):
    def_path = Path(__file__).parent
    file_name= f"default.yaml" if not args[1:] else args[1]
    if not file_name.startswith("/"):
        file_name = f"{def_path}/{file_name}"
    yaml_file = open(file_name, 'r')
    datas = yaml.load_all(yaml_file, Loader=yaml.FullLoader)
    return datas


def main(datas):
    lang = locale.getdefaultlocale()[0].lower()[0:2]
    for actions in datas:
        for key, action in actions.items():
            print("\n")
            print("-" * 12, key, "-" *12)
            # print(f"{key} -> {action}")
            try:
                print("::", action["title"][lang])
            except KeyError:
                try:
                    print("::", action["title"]["en"])
                except KeyError:
                    pass
            try:
                print("\t", action["command"])
            except KeyError:
                pass
            try:
                if requires := action["require"]:
                    not_run = False
                    for require in requires:
                        if str(require).startswith("/"):
                            if not Path(require).exists():
                                print(f"Error: file {require} not found")
                                not_run = True
                        else:
                            # package installed ?
                            require = require.lower()
                            ret = [ True for p in pamac_db.get_installed_pkgs() if p.get_name() == require]
                            if not ret:
                                print(f"Error: package {require} not installed")
                                not_run = True
                    if not_run:
                        continue
            except KeyError:
                pass
            try:
                action_type = action["type"]
            except KeyError:
                action_type = "shell"
            print("-" * 44, "\n")
            if action_type == "shell":
                with subprocess.Popen([f"env TERM=xterm LANG=C {action['command']}"], universal_newlines=True, stdout=subprocess.PIPE, shell=True, text=True) as process:
                    print(process.stdout.read())
            if action_type == "include":
                if action["object"] == "journald":
                    try:
                        level = action["level"]
                    except KeyError:
                        level = 3
                    journald = Journald(level)
                    journald.run()
                    journald.print()


if __name__ == "__main__":
    datas = load_yaml(sys.argv)
    os.environ['TERM'] = "xterm"
    main(datas)
