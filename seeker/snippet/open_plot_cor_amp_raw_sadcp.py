#date: 2023-05-11T16:47:47Z
#url: https://api.github.com/gists/1d1daec0eaf089c465ab2cec9486c5aa
#owner: https://api.github.com/users/hhourston

# Use in mtgk venv with python3

from pycurrents.adcp.rdiraw import rawfile, Multiread
import matplotlib.pyplot as plt
import numpy as np
import glob
from datetime import datetime
import pandas as pd

year = '2018'
cruise_num = '40'
input_dir = '/home/hourstonh/Documents/adcp_processing/ship_mounted/2018-040_LineP/'
input_files = glob.glob(input_dir + '*.ENS')
input_files.sort()
# input_files.remove('/home/hourstonh/Documents/adcp_processing/ship_mounted/iml4_2017_sw_01.ENS')

# input_file = 'ADCP_20220810T211249_148_000000.ENS'
# fig_name = input_file.replace('.ENS', '_amp1_cor1.png')

# data = rawfile(input_file, 'os', trim=True)

data = Multiread(input_files, 'os', yearbase=2022)

fixed_leader = data.read(varlist=['FixedLeader'])
vel = data.read(varlist=['Velocity'])
amp = data.read(varlist=['Intensity'])
cor = data.read(varlist=['Correlation'])
pg = data.read(varlist=['PercentGood'])

# Calculate some variables

# Distance dimension
bin_distances = vel.dep
bin_depths = np.nanmean(
    vel.VL['XducerDepth']/10) + bin_distances

# Convert time from decimal years to pandas datetime
# data_origin = pd.to_datetime(vel.dday[0], unit='D',
#                              origin=year + '-01-01',
#                              utc=True)
data_origin = pd.Timestamp(year + '-01-01')
dtime = pd.to_datetime(
    vel.dday, unit='D', origin=data_origin,
    utc=True).to_numpy()
# Time since start time
time_since_st = np.array([t - dtime[0] for t in dtime])

#dhms = [pd.Timedelta(d, unit='day') + pd.Timedelta(h, unit='h') +
#        pd.Timedelta(m, unit='minute') + pd.Timedelta(s, unit='s') +
#        pd.Timedelta(ns, unit='ns') for d,h,m,s,ns in
#        zip(dtime.day, dtime.hour, dtime.minute, dtime.second,
#            dtime.nanosecond)]

# Plot data
beam_num = 4


fig_name = 'ADCP_{}-{}_beam_{}_amp_cor.png'.format(
    year, cruise_num, beam_num)

fig = plt.figure()

# Subset out the bad data for 2018-026 cruise
if str(year) == '2018' and str(cruise_num) == '26':
    dtime_diffs = np.diff(dtime)
    max_diff_idx = np.where(dtime_diffs == max(dtime_diffs))[0][0] + 3
    #time_since_st = np.array([t - dtime[max_diff_idx] for t in dtime])
    fig_name = 'ADCP_{}-{}_beam_{}_amp_cor_subset.png'.format(
    year, cruise_num, beam_num)

# Plot correlation
ax = fig.add_subplot(2,1,1)
f1 = ax.pcolormesh(dtime, bin_depths,
                   cor['cor{}'.format(beam_num)].T,
                   shading='auto')
#f1 = ax.pcolormesh(dtime[max_diff_idx:], bin_depths,
#                   cor['cor{}'.format(beam_num)][max_diff_idx:,:].T,
#                   shading='auto')
cbar = fig.colorbar(f1)
ax.set_title('Correlation')
ax.set_ylabel('Depth [m]')
xticks = ax.get_xticks()
ax.set_xticks(xticks[::2])

plt.gca().invert_yaxis()

# Plot amplitude
ax2 = fig.add_subplot(2, 1, 2)
f2 = ax2.pcolormesh(dtime, bin_depths,
                    amp['amp{}'.format(beam_num)].T,
                    shading='auto')
#f2 = ax2.pcolormesh(dtime[max_diff_idx:], bin_depths,
#                    amp['amp{}'.format(beam_num)][max_diff_idx:,:].T,
#                    shading='auto')
cbar = fig.colorbar(f2)
ax2.set_title('Amplitude')
ax2.set_ylabel('Depth [m]')
# ax2.set_xlabel('Time since {} UTC'.format(dtime[0]))
ax2.set_xlabel('Time [UTC]')
xticks = ax2.get_xticks()
ax2.set_xticks(xticks[::2])

plt.gca().invert_yaxis()
plt.suptitle('Beam {}'.format(beam_num))

# Reduce white space around plots
plt.tight_layout()

plt.savefig(fig_name)
plt.close(fig)
