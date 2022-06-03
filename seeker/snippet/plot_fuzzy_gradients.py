#date: 2022-06-03T17:06:18Z
#url: https://api.github.com/gists/cc275226b6e8ef82febe7958df6adeca
#owner: https://api.github.com/users/NikolayOskolkov

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

species1 = list(block_df.iloc[:,0:99].mean(axis = 1))
species2 = list(block_df.iloc[:,100:199].mean(axis = 1))

plt.plot(range(0,200,1), species1, color = 'blue')
plt.plot(range(0,200,1), species2, color = 'orange')
plt.ylabel('SPECIES ABUNDANCE', fontsize = 22)
plt.xlabel('SAMPLES', fontsize = 22)

my_legends = [mpatches.Patch(color = 'blue', label = 'Species 1'), 
              mpatches.Patch(color = 'orange', label = 'Species 2')]
plt.legend(handles = my_legends, fontsize = 20)