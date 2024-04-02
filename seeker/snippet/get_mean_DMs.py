#date: 2024-04-02T17:04:17Z
#url: https://api.github.com/gists/f16e9f34d2ab7150debc374c9d27a633
#owner: https://api.github.com/users/vivekvenkris

import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat

'''
Reads Paulo's GCpsr.txt file and calculates the mean DM for each pulsar. Assumes that the DM is in the 5th column
Removes all empty lines and lines starting with #. Also skips lines where the DM is marked with a *
If the DM has an error, it is assumed to be in the form of DM (error) and the error is used to calculate the mean error
If the DM has no error, the error is assumed to be 0.0
Any line that does not start with J or B is considered to start a new cluster, and hence cluster name is updated
Change print_error to True if you want to print the error as well
'''


DM_COLUMN = 4;
print_error = False

with open('/Users/vkrishnan/trashcan/GCpsr.txt') as f:
    data = f.readlines()

# dictionary of list of DMS
DMS = {}
cluster_name = "NONE"
for d in data:
    if "#" in d or d.strip() == "":
        continue

    chunks = d.split()
    if "J" not in chunks[0] and "B" not in chunks[0]:
        cluster_name = d.strip()
        continue

    if chunks[DM_COLUMN] == "*":
        print("Skipping ", d)
        continue

    try:
        dm_str = chunks[DM_COLUMN]
        if "(" in dm_str:
            dm_val = float(dm_str.split("(")[0])
            dm_err = float(dm_str.split("(")[1].split(")")[0])
            dm = ufloat(dm_val, dm_err)
        else:
            dm = ufloat(float(dm_str), 0.0)
        if cluster_name in DMS:
            DMS[cluster_name].append(dm)
        else:
            DMS[cluster_name] = [dm]
    except ValueError:
        print("Skipping invalid format line: ", d)

for k in DMS:
    dms = DMS[k]
    mean_dm = np.mean(dms)
    mean = mean_dm.nominal_value
    std = mean_dm.std_dev
    mean = "{:.03f}".format(mean)
    std = "{:.03f}".format(std)
    if print_error:
        print(k, mean, std)
    else:
        print(k, mean)





   
    