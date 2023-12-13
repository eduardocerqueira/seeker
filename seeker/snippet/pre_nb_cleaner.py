#date: 2023-12-13T16:58:32Z
#url: https://api.github.com/gists/ae9399f61a5a4776576963e89caedf39
#owner: https://api.github.com/users/Andreal2000

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json

tmp_extension = "tmp"
remove_metadata_fields = ["collapsed", "scrolled"]

current_dir = os.getcwd()

list_of_files = []
for (dirpath, dirnames, filenames) in os.walk(current_dir):
    list_of_files += [os.path.join(dirpath, file) for file in filenames]

list_of_ipynb_files = [file for file in list_of_files if file.endswith(".ipynb")]

for file in list_of_ipynb_files:
    ipynb_file = json.loads(open(file, "r").read())
    for cell in ipynb_file["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
            # Remove metadata associated with output
            if "metadata" in cell:
                for field in remove_metadata_fields:
                    cell["metadata"].pop(field, None)

    print(f"Cleared notebook: {file}")
    os.rename(file, f"{file}.{tmp_extension}")
    json.dump(ipynb_file, open(file, "w"))

print("All notebooks have been cleared of their outputs.")
