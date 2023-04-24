#date: 2023-04-24T16:52:16Z
#url: https://api.github.com/gists/f63d52bd31b4e372d2314450bf21210b
#owner: https://api.github.com/users/abhishekkassetty

import os
import os.path
import shutil


destination = "give destination path"
for root, dir, files in os.walk('give source path'):
    for ffile in files:
        if os.path.splitext(ffile)[1] in ('.mp4'):
            src = os.path.join(root, ffile)
            shutil.copy(src, destination)
            
            