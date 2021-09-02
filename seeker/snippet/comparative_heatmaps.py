#date: 2021-09-02T17:11:36Z
#url: https://api.github.com/gists/348942442ae089f3290503b98292d6ac
#owner: https://api.github.com/users/betacosine

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import seaborn as sns

df = pd.read_csv('read/from/directory/file.csv')

#subset the dataframe to include only the column you want to compare on the y-axis
#and the columns with metrics to compare on

df_sub = df[['Organization Name','Education','Corruption','Consumer','Governance','Information','Gender',
                  'HumanRights','Occupational','Taxes','Env.','Work Cond.'
df_sub = df_sub.set_index('Organization Name')

figure, ax = plt.subplots(figsize=(10,10))
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
ax.set_xticklabels(list(df_sub.columns), size=14)
ax.set_yticklabels(list(df_sub.index.values), size=14)
#sns.set(font_scale=5)
svm = sns.heatmap(df_sub, mask=(df_sub==0), center=0, cmap='Greens', cbar=False, 
                  xticklabels=True, yticklabels=True, linewidths=.1,ax=ax)

figure = svm.get_figure()    
figure.savefig(in_file+'save_heatmap.png', orientation='landscape',
               transparent=True,format='png', pad_inches=.1)
