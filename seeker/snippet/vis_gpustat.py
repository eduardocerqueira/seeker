#date: 2021-12-08T17:13:55Z
#url: https://api.github.com/gists/7eb7c9da98495b3c162aa16399e4e041
#owner: https://api.github.com/users/olegkhomenko

import gpustat
import tqdm
import time
import seaborn as sns
import numpy as np
import pandas as pd

# Query for 6 seconds (step is 0.1 second)
gpu_stats = []
for i in range(60):
    time.sleep(0.1)
    gpu_stats += [gpustat.GPUStatCollection.new_query()]

# Prepare time series
gpu0 = [g[0].utilization for g in gpu_stats]
gpu1 = [g[1].utilization for g in gpu_stats]
gpu2 = [g[2].utilization for g in gpu_stats]
gpu3 = [g[3].utilization for g in gpu_stats]

# Prepare plot names
gpu_names = ["|".join([p['username'] for p in n.processes]) for n in gpu_stats[0]]

# Plots
df = pd.DataFrame([gpu0, gpu1, gpu2, gpu3]).T
df.columns = gpu_names
df.plot()