#date: 2025-02-07T16:55:02Z
#url: https://api.github.com/gists/ba93887c08f997b182ca9998a53826be
#owner: https://api.github.com/users/EvgeniGordeev

import re
import subprocess

if __name__ == '__main__':
    # located in project folder .idea/dataSources.xml
    with open('dataSources.xml', 'r') as f:
        data = f.read()
        data_sources = sorted(re.findall(r".*data-source.*name=\"(.*?)\".*uuid=\"(.*?)\".*", data))
        for name, uuid in data_sources:
            command = "**********"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            print(f"{name}={result.stdout.strip()}")xt=True)
            print(f"{name}={result.stdout.strip()}")