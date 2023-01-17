#date: 2023-01-17T17:06:42Z
#url: https://api.github.com/gists/7cca9debff0e172483fdd0c13a217ae3
#owner: https://api.github.com/users/zabid

# TQDM: Python progress bar tutorial
# Youtube Video: https://www.youtube.com/watch?v=n4E7of9BINo

import pandas as pd
import numpy as np
from time import sleep
from tqdm import tqdm, trange

# Create Fake Data
dogs = np.random.choice(['labradoodle','beagle','mutt'], size=50_000)
smell = np.random.randint(0, 100, size=50_000)
df = pd.DataFrame(data=np.array([dogs, smell]).T,
                  columns=['dog','smell'])

# Basics
for dog in tqdm(dogs):
    sleep(0.000001)
    break
print('done')
    
#trange example
for i in trange(50, ncols=55):
    sleep(0.1)
print('done')

# Manually entering total length
for dog in tqdm(dogs, total=50_001):
    sleep(0.00001)

# Create a pbar
pbar = tqdm(total=50_000)
for s in smell:
    pbar.update(1)
    sleep(0.00001)
pbar.close()

# Setting description and unit
for dog in tqdm(dogs, desc='dog counter', unit='barks'):
    sleep(0.00001)

# Nested progress bars
for dog in tqdm(dogs[:5], desc='dog counter', total=5):
    for s in tqdm(smell[:2], desc='smell counter', total=2):
        sleep(0.1)

# Dynamic description
pbar = tqdm(dogs[:10])
for dog in pbar:
    sleep(0.5)
    pbar.set_description(f'Procressing {dog}')

# Control the Bar Size
for i in tqdm(range(9999999), ncols=55):
    pass

# Remove the time stats
for i in tqdm(range(9999999), ncols=20):
    pass

# Only show %
for i in tqdm(range(9999999), ncols=4):
    pass

# Setting the interval
for dog in tqdm(dogs, mininterval=1):
    sleep(0.00001)

for dog in tqdm(dogs, maxinterval=100):
    sleep(0.00001)
    
# Disabiling tqdm
debug = False
for s in tqdm(smell, disable=not debug):
    sleep(0.00001)

# TQDM and Pandas
tqdm.pandas(desc='dog bar')
out = df.progress_apply(lambda row: row['smell']**2, axis=1)

# TQDM Notebook
from tqdm.notebook import tqdm
for dog in tqdm(dogs):
    sleep(0.000001)
    
# Notebook bar turns red if breaks
counter = 0
for dog in tqdm(dogs):
    if dog == "beagle":
        counter += 1
    if counter == 10_000:
        break
        
# Tqdm auto (selects tqdm or tqdm.notebook version)
from tqdm.auto import tqdm

# Tqdm in the command line
!seq 9999999 | python3 -m tqdm --bytes | wc -l

# Read the docs:
help(tqdm)