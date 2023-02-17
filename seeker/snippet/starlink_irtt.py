#date: 2023-02-17T17:06:04Z
#url: https://api.github.com/gists/15ebbbdb739ffdb72b1d1687ca5b68b1
#owner: https://api.github.com/users/virtuallynathan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, FixedLocator
import math
import json
import zipfile


name = 'starlink_pm_jan_23_2023'

zip_file = f'starlink_data/{name}.json.zip'
file_to_extract = f'{name}.json'

 
try:
    with zipfile.ZipFile(zip_file) as z:
        with open(file_to_extract, 'wb') as f:
            f.write(z.read(file_to_extract))
            print("Extracted", file_to_extract)
except:
    print("Invalid file")
    
f = open(f'{name}.json')
data = json.load(f)

round_trips = data['round_trips']
rtts = []
ts = []
index = []
color = []
count = 0
for round_trip in round_trips:
    ts.append(round_trip['timestamps']['client']['send']['wall'])
    if round_trip['lost'] == 'false':
        rtts.append(round_trip['delay']['rtt']/1000000)
    else:
        rtts.append(-1)
    if count % 2 == 1:
        color.append("red")
    else: 
        color.append("blue")
    index.append(count)
    count = count + 1
df = pd.DataFrame()
df['rtts'] = rtts
df['ts'] = ts
df['color'] = color
df['date'] = df['ts'].astype('datetime64[ns]')

resampled = df.set_index('date').resample('15S', offset='12S').apply(lambda x: (x==-1.0).sum()/len(x)*100)

resampled1 = df[df.rtts != -1.0].set_index('date').resample('15S', offset='12S').agg(['mean', 'median', 'std', lambda x: x.quantile(0.99)])
resampled1
#df['mean_non_negative'] = resampled['column_name']['mean']
#df['median_non_negative'] = resampled['column_name']['median']
#df['std_non_negative'] = resampled['column_name']['std']

plt.figure(figsize=(450, 15))
plt.scatter(df['date'], df['rtts'], c=df['color'])
plt.title('Dishy RTT')
plt.xlabel('Time')
plt.ylabel('RTT (ms)')
plt.xticks(rotation=45)
plt.grid()
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.SecondLocator(bysecond=[12, 27, 42, 57]))
for i, row in resampled.iterrows():
    plt.annotate(f"Slot Loss: {row['rtts']:.1f}%", 
                 xy=(i, row['rtts']), 
                 xytext=(6, 3),
                 textcoords='offset points',
                 fontsize=14,
                 arrowprops=dict(arrowstyle="->",
                                 connectionstyle="arc3,rad=.2"))
    
for i, row in resampled1.iterrows():
    plt.annotate(f"Median: {row['rtts']['median']:.1f}ms \n Std Dev.:  {row['rtts']['std']:.1f}ms", 
                 xy=(i, row['rtts']['median']), 
                 xytext=(5, 350),
                 textcoords='offset points',
                 fontsize=14,
                 arrowprops=dict(arrowstyle="->",
                                 connectionstyle="arc3,rad=.2"))
    
plt.savefig(f'starlink_data/{name}.pdf')
plt.show()