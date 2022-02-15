#date: 2022-02-15T17:07:34Z
#url: https://api.github.com/gists/c4674be237592460681cb7b361746510
#owner: https://api.github.com/users/arjunsavel

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import pandas as pd

##### CHANGE this to your absolute file path to your EOS file! #####
eos_filename = 'RT_3D_Transmission_Code/Data/eos_solar_gas_full_hires.dat'

##### CHANGE this to your absolute file path to your GCM file! #####
gcm_filename = 'RT_3D_Transmission_Code/Data/t_p_profiles/t_p_3D_WASP76_deep_rcb_1e7.dat'
frame = pd.read_csv(gcm_filename,
                         delim_whitespace=True,
                         names=('lat', 'lon', 'level', 'alt', 'pres',
                                'temp', 'u', 'v'))

eos = pd.read_csv(eos_filename, delim_whitespace=True).dropna()


##### CHANGE the last column if you'd like to make different maps! #####
f = interp2d(eos['T'], eos['P'], eos['OH'])
f2 = interp2d(eos['T'], eos['P'], eos['HCN'])


OHs = [] 
HCNs = [] 

for i in tqdm(range(len(frame))):
    
    temp = frame['temp'].iloc[i]
    if temp <= 0.0:
        OH = 0
        HCN = 0
    else:
        OH = f(frame['temp'].iloc[i], frame['pres'].iloc[i])[0]
        HCN = f2(frame['temp'].iloc[i], frame['pres'].iloc[i])[0]
    OHs += [OH]
    HCNs += [HCN]
    
frame.loc[:, 'OH'] = OHs
frame.loc[:, 'HCN'] = HCNs


#### CHANGE these if you want to plot different levels! ####
single_level = frame[frame.level==100]

single_level2 = frame[frame.level==75]

single_level3 = frame[frame.level==50]


#### CHANGE these if you have different lat / lon grid!

nlat = 94
nlon = 192

OH_plot = np.log10(single_level.OH.values.reshape(nlat, nlon))
OH_plot2 = np.log10(single_level2.OH.values.reshape(nlat, nlon))
OH_plot3 = np.log10(single_level3.OH.values.reshape(nlat, nlon))

HCN_plot = np.log10(single_level.HCN.values.reshape(nlat, nlon))
HCN_plot2 = np.log10(single_level2.HCN.values.reshape(nlat, nlon))
HCN_plot3 = np.log10(single_level3.HCN.values.reshape(nlat, nlon))

vmin = np.min([OH_plot, OH_plot2, OH_plot3, HCN_plot, HCN_plot2, HCN_plot3])
vmax = np.max([OH_plot, OH_plot2, OH_plot3, HCN_plot, HCN_plot2, HCN_plot3])

fig, axs = plt.subplots(3, 2, figsize=(9,9), sharex=True, sharey=True)

temp_min = frame.temp.min()
temp_max = frame.temp.max()

# plt.figure()

axs[0,0].set_title(r'$\rm log_{10}$(OH) field', fontsize=13)
axs[0,1].set_title(r'$\rm log_{10}$(HCN) field', fontsize=13)
m = axs[0,0].imshow(OH_plot, vmin=vmin, vmax=vmax, cmap='magma')
axs[0,1].imshow(HCN_plot, vmin=vmin, vmax=vmax, cmap='magma')

axs[0,0].text(10,80, r'P$\approx$1 bar', fontsize=16)

m = axs[1,0].imshow(OH_plot2,  vmin=vmin, vmax=vmax, cmap='magma')
axs[1,0].text(10,80, r'P$\approx$30 mbar', fontsize=16)
axs[1,1].imshow(HCN_plot2, vmin=vmin, vmax=vmax, cmap='magma')

m = axs[2,0].imshow(OH_plot3,  vmin=vmin, vmax=vmax, cmap='magma')
axs[2,0].text(10,80, r'P$\approx$0.7 mbar', fontsize=16)
axs[2,1].imshow(HCN_plot3,vmin=vmin, vmax=vmax,  cmap='magma')

cb_ax = fig.add_axes([.95, 0.147, 0.03, 0.708])
cbar = fig.colorbar(m, cax=cb_ax)

#  set the colorbar ticks and tick labels
# cbar.set_ticks(np.arange(0, 1.1, 0.5))
cbar.set_ticklabels(['low', 'medium', 'high'])
cbar.set_label('Temperature (K)', fontsize=20)

for ax in axs.flatten():
    y_label_list = [-90, 0, 90]

    ax.set_yticks([0, 47, 94])

    ax.set_yticklabels(y_label_list)
    
    x_label_list = [0, 90, 180, 270, 360]

    ax.set_xticks([0, 48, 96, 144, 192])

    ax.set_xticklabels(x_label_list)

thing = fig.add_subplot(111, frameon=False)


plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Longitude (degrees)', fontsize=20, labelpad=10)
plt.ylabel('Latitude (degrees)', fontsize=20, labelpad=10)