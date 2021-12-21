#date: 2021-12-21T17:08:45Z
#url: https://api.github.com/gists/bd75c469559f03596bd2d274dfb5a315
#owner: https://api.github.com/users/jreadey

import random
import sys
import h5pyd
import numpy as np


#
# main
# 
if len(sys.argv) <= 1 or sys.argv[1] in ("-h", "--help"):
    print("Perf test for retreiving data from nsrdb data")
    print("")
    print("usage: python nsrdb_test.py <filepath> [--bucket bucket_name] [--dataset dataset] [--fancy] [--count count]")
    print("example: python nsrdb_test.py /nrel/nsrdb/v3/nsrdb_2000.h5")
    print("or with bucket arg: python perf_test.py /nrel/nsrdb/v3/nsrdb_2000.h5 --bucket nrel-pds-hsds")
    print("or with fancy selection: perf_test.py /nrel/nsrdb/v3/nsrdb_2000.h5 --bucket nrel-pds-hsds --fancy")
    print("Note: fancy selection requires h5pyd version 0.9.2 or higher")
    print("by default the 'wind_speed' dataset will be used.  Use the --dataset option with")
    print("dataset name to use another dattaset")
    print("The default number of columns retrieved will be 1000; use --count to adjust")
    sys.exit(0)

filepath = sys.argv[1]
bucket = None
dataset_name = "wind_speed"
use_fancy = False
coord_count = 1000
print("filepath:", filepath)
bucket_option = False
dataset_option = False
count_option = False
for i in range(2, len(sys.argv)):
    arg = sys.argv[i]
    if bucket_option:
        bucket = arg
        bucket_option = False
    elif dataset_option:
        dataset_name = arg
        dataset_option = False
    elif count_option:
        coord_count = int(arg)
        count_option = False
    elif arg == "--bucket":
        bucket_option = True
    elif arg == "--dataset":
        dataset_option = True
    elif arg == "--count":
        count_option = True
    elif arg == "--fancy":
        use_fancy = True
    else:
        print(f"Unexpected option: {arg}")
        sys.exit(1)
    
print("filepath:", filepath)
print("bucket:", bucket)
print("dataset:", dataset_name)
print("use_fancy:", use_fancy)
print("count:", coord_count)
f = h5pyd.File(filepath, "r", use_cache=False, bucket=bucket)
dset = f[dataset_name]
print("dset:", dset, dset.id.id)
if len(dset.shape) != 2:
    print("datset must be of rank 2")
    sys.exit(1)
if dset.shape[1] < coord_count:
    print("dataset second dimension must be greater than count")
    sys.exit(1)
print("chunks:", dset.chunks)
 
# choose some random coordinates
coords = []
while len(coords) < coord_count:
    n = random.randint(0,2975)
    if n not in coords:
        coords.append(n)
coords.sort()

# retreive data 
if use_fancy:
    # do one fancy selection request
    arr = dset[:, coords]
else:
    # get data column by column
    arr = np.zeros((dset.shape[0], coord_count), dtype='i2')
    for i in range(coord_count):
        index = coords[i]
        arr1d = dset[:, index]
        arr[:,i] = arr1d

print(f"arr_min: {arr.min()}, arr_max:{arr.max()}, arr_mean: {arr.mean():0.2f}")