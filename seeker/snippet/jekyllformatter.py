#date: 2025-04-22T16:45:47Z
#url: https://api.github.com/gists/da5266c7664a5448f88bdd6b80153f43
#owner: https://api.github.com/users/daisyUniverse

# Jekyll Formatter
# Converts normal .MD files into Jekyll formatted files
# Daisy Universe [D]
# 04 . 22 . 25

import os.path 
import time 
from datetime import datetime
from pathlib import Path

src     = Path("/mnt/blogs/")
dest    = "/home/daisy/blogsites/"

pathlist = src.glob('**/*.md')
for path in pathlist:
    
    # Generate a new Jekyll friendly name based on the file creation date
    pathstr     = str(path)
    timeCreated = datetime.fromtimestamp(os.path.getctime(pathstr))
    dateCreated = str(datetime.strftime(timeCreated, "%Y-%m-%d")).split(" ")[0]
    newname     = ( dateCreated + "-" + pathstr.split("/")[-1].replace(" ","-") )
    section     = pathstr.split("/")[-2]

    print( "[" + section + "] " + newname)
    print("SOURCE: " + pathstr)

    # Read the file and modify it to be Jekyll friendly
    title   = pathstr.split("/")[-1].replace(".md", "")
    with open(pathstr, "r+") as fp:
        lines = fp.readlines()
        lines.insert(0, "---\n\n")
        lines.insert(0, "layout: post\n")
        lines.insert(0, ("title: " + title + "\n"))
        lines.insert(0, "---\n")
        fp.seek(0)
        output = lines

    # Write new data to target path
    newpath = ( dest + section + "/_posts/" + newname)
    print("DESTINATION: " + newpath + "\n")

    with open(newpath, "w+") as fp:
        fp.writelines(output)