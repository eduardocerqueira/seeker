#date: 2025-07-14T17:11:41Z
#url: https://api.github.com/gists/feb53b0396a6fba58b29894f961650be
#owner: https://api.github.com/users/PierceLBrooks


# Author: Pierce Brooks

import os
import shutil
import subprocess

for root, folders, files in os.walk(os.getcwd()):
    for name in files:
        path = os.path.join(root, name)
        command = []
        command.append("exiftool")
        command.append(path)
        language = ""
        try:
            output = subprocess.check_output(command)
            output = str(output.decode("UTF-8"))
            lines = output.split("\n")
            for line in lines:
                line = line.strip()
                if (line.startswith("Media Language Code")):
                    line = line.split(":")
                    language += line[len(line)-1].strip().replace("/", "_").replace(";", "_").replace(":", "_")
                    break
        except:
            language = ""
        if (len(language) == 0):
            language += "und"
        os.makedirs(os.path.join(os.getcwd(), language), exist_ok=True)
        shutil.copy2(path, os.path.join(os.getcwd(), language, name))
    break
