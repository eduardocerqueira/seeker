#date: 2023-12-13T16:58:32Z
#url: https://api.github.com/gists/ae9399f61a5a4776576963e89caedf39
#owner: https://api.github.com/users/Andreal2000

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

tmp_extension = "tmp"

current_dir = os.getcwd()

list_of_files = []
for (dirpath, dirnames, filenames) in os.walk(current_dir):
    list_of_files += [os.path.join(dirpath, file) for file in filenames]

list_of_ipynb_files = [file for file in list_of_files if file.endswith(f".ipynb.{tmp_extension}")]

for file in list_of_ipynb_files:
    print(f"Restored notebook: {file}")
    os.replace(file, file.replace(f".{tmp_extension}", ""))
    
print("All notebooks have been restored to their original names.")
