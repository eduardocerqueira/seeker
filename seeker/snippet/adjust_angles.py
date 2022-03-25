#date: 2022-03-25T17:08:42Z
#url: https://api.github.com/gists/e47f6ae3e8101fe0e9a47e6a81579c70
#owner: https://api.github.com/users/bbarad

# Adjust angles from imod alignment before warp import to allow flattening of tomograms
# Replaces imod's taSolution.log file with adjusted deltlts, while preserving the original file in taSolution.log.bak
# Author: Benjamin Barad 2022
# Usage:
# cd WARPTOPFOLDER/imod
# python adjust_angles.py * 
# Alternatively, instead of the wildcard individual tilt series folders can be specified.


import pandas as pd
from sys import argv
from os import path
import shutil

folders = argv[1:]
tilt_offset = 11
file_header = ["view","rotation","tilt","deltilt","mag","dmag","skew","mean resid"]

for folder in folders:
    filename = folder+"/taSolution.log"
    if not path.exists(filename):
        print("File not found:",filename)
        continue
    if not path.exists(filename+".bak"):
        print("Saving backup file:",filename+".bak")
        shutil.copyfile(filename,filename+".bak")
    else:
        print("Backup file already exists:",filename+".bak")
    df = pd.read_csv(filename, index_col=0, skiprows=3, sep=None, skipinitialspace=True,header=None, names=file_header)
    df["deltilt"] = df["deltilt"]+tilt_offset
    df.to_csv(filename, sep=" ", index=True, header=True)