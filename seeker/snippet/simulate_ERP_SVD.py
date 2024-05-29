#date: 2024-05-29T16:40:57Z
#url: https://api.github.com/gists/cdec35e64ea3f8c3ef31c90187fb09a3
#owner: https://api.github.com/users/mdnunez

# Copyright 2024 Michael D. Nunez
#
# MIT License
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Possible citation for SVD method to calculate single-trial ERPs:

# Nunez, M. D., Vandekerckhove, J., & Srinivasan, R. (2017). How attention 
# influences perceptual decision making: Single-trial EEG correlates of 
# drift-diffusion model parameters. Journal of mathematical psychology, 76, 
# 117-130. doi.org/10.1016/j.jmp.2016.03.003

# Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# 29/05/24      Michael D. Nunez                          Original code

import os
import numpy as np
import mne
from mne.datasets import fetch_fsaverage
from mne import make_ad_hoc_cov
import matplotlib.pyplot as plt


# Fetch fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = fs_dir
subject= 'fsaverage'
trans = 'fsaverage'  # Using 'fsaverage' as the transformation
src = fs_dir + '/bem/fsaverage-ico-5-src.fif'  # Surface-based source space
bem = fs_dir + '/bem/fsaverage-5120-5120-5120-bem-sol.fif'  # BEM solution

# Set up the standard 64-channel BioSemi montage
montage = mne.channels.make_standard_montage('biosemi64')
info = mne.create_info(montage.ch_names, sfreq=256, ch_types='eeg')
info.set_montage(montage)


forward_file = 'fsaverage-ico-bem-biosemi64-fwd.fif'
if not os.path.exists(forward_file):
    # Make the forward model
    temp_fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem, 
        eeg=True, mindist=5.0, n_jobs=1)
    fwd = mne.convert_forward_solution(temp_fwd, force_fixed=True)
    # Save the forward model
    mne.write_forward_solution(forward_file, fwd)
else:
    # Load the forward model
    print('Loading forward model...')
    temp_fwd = mne.read_forward_solution(forward_file)
    fwd = mne.convert_forward_solution(temp_fwd, force_fixed=True)

# Generate one source time course
time_secs = 1  # Time window in seconds
time_samples = int(np.round(info['sfreq']*time_secs)) # Time window in samples
time_step = 1.0/info['sfreq']
freq = 5  # Frequency in Hz
time = np.linspace(0, time_secs, num=time_samples)
time_shift = 0 # In seconds
sample_shift = info['sfreq']*time_shift
scaler = 2.5e-10 # Scaled to a reasonable source amplitude
dampen = 5 # See lambda of exponential distribution
source_waveform = np.sin(2 * np.pi * freq * time - sample_shift
) * np.exp(-dampen*time) * scaler # Simulate activity as dampening sine wave
plt.figure()
plt.plot(time,source_waveform)
plt.savefig('Simulated_source_waveform.png')
plt.close('all')

# Select a region to activate: the right and left inferior parietal cortices
# print('Labels to choose from...')
# print(mne.read_labels_from_annot(subject))
selected_labels = mne.read_labels_from_annot(subject, 
    regexp='superiorparietal-\wh')
location = "center"  # Use the center of the region as a seed.
extent = 10.0  # Extent in mm of the region.
extended_labels_lh = mne.label.select_sources(subject, selected_labels[0], 
    location=location, extent=extent)
extended_labels_rh = mne.label.select_sources(subject, selected_labels[1], 
    location=location, extent=extent)

# Define when the activity occurs
n_events = 100
events = np.zeros((n_events, 3), int)
events[:, 0] = 0 + time_samples*np.arange(n_events)  # Events sample.
events[:, 2] = 1  # All events have the id 1.

# Create a SourceSimulator object
source_simulator = mne.simulation.SourceSimulator(fwd['src'], tstep=time_step)

# Add the simulated activity to the SourceSimulator, left hemisphere
source_simulator.add_data(extended_labels_lh, source_waveform, events)
# Add the simulated activity to the SourceSimulator, right hemisphere
source_simulator.add_data(extended_labels_rh, source_waveform, events)

# Simulate the EEG data without noise
raw = mne.simulation.simulate_raw(info, source_simulator, forward=fwd, 
    n_jobs=4)

# Add noise to the simulation
cov = mne.make_ad_hoc_cov(raw.info)
mne.simulation.add_noise(raw, cov, iir_filter=[0.2, -0.2, 0.04])
raw.set_eeg_reference(ref_channels='average') # Set to average reference
print(raw.info)

# Dimensions of the simulated EEG data
print(raw._data.shape)
total_sim_record_time = raw._data.shape[1]/raw.info['sfreq']
print(f'The simulated recording time was {total_sim_record_time} seconds')

# Plot the simulated EEG data
# raw.plot()

plot_electrodes = ['Fpz', 'AFz', 'FCz', 'Fz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz', 
	'Iz']
central_data = raw.get_data(plot_electrodes, stop=time_samples)
plt.figure()
plt.plot(time,central_data.T)
plt.legend(plot_electrodes)
plt.savefig('Simulated_first_trial_central_electrodes.png')
plt.close('all')

# PLOT ERPs

# Filter the data
filt_raw = raw.copy() 
filt_raw.load_data().filter(l_freq = 0.1, h_freq = 40)

# Epoch the data
epochs = mne.Epochs(filt_raw,
                    events,
                    event_id = {'Trial Onset':1},
                    tmin = 1/raw.info['sfreq'], tmax = 1,
                    proj = False, baseline = None,
                    preload = True)
epochs.events
epoched_data = epochs.get_data()
epoched_data.shape

# Average all of the trials for a grand-average
evoked = epochs.average()

plt.ion() # Turn on interactive mode and off matplotlib blocking
fig = evoked.plot()
fig.savefig('Simulated_all_electrodes_ERP.png')
plt.close('all')

fig = evoked.plot_joint()
fig.savefig('Simulated_all_electrodes_ERP_peaks.png')
plt.close('all')
plt.ioff() # Turn off interactive mode

# EPOCH DATA (without relying on MNE)

# Find the event IDs of interest
# plot_data = raw._data
plot_data = raw.get_data()
ntrials = n_events
nchans = plot_data.shape[0]
nsamps_per_epoch = int(plot_data.shape[1]/ntrials)
epoched_self = np.reshape(plot_data,(nchans,nsamps_per_epoch,ntrials),
    order='F')
eegdata = np.transpose(epoched_self, (1, 0, 2))
# Test reshaping
assert(plot_data[0,255] == eegdata[255,0,0])
assert(plot_data[9,255] == eegdata[255,9,0])
assert(plot_data[9,255+256*3] == eegdata[255,9,3])
assert(plot_data[7,123+256*49] == eegdata[123,7,49])

# EXTRACT WEIGHTS FROM SINGULAR VALUE DECOMPOSITION (SVD) ON TRIAL AVERAGE

# Calculate ERP matrix
erp_matrix = np.mean(eegdata, axis=2)

u, s, v = np.linalg.svd(erp_matrix, full_matrices=0)

# See the matrix algebra in the second input of the follow assert()
assert(np.allclose(erp_matrix, u * s @ v))

component1_weights = v[0,:]

# Percentage variance explained
per_exp = ( np.square(s) / float(np.sum(np.square(s))) )*100

print(f'Percentage variance explained by the first component {per_exp[0]:.1f}%')

plt.ion() # Turn on interactive mode and off matplotlib blocking
mne.viz.plot_topomap(component1_weights,info)
plt.savefig('Component1_weights.png',dpi=300)
plt.close('all')
plt.ioff() # Turn off interactive mode

first_trial_eeg = raw.get_data(stop=time_samples)
plt.figure()
plt.plot(time,first_trial_eeg.T @ component1_weights)
plt.savefig('Simulated_first_trial_component1.png')
plt.close('all')

# Compute all single-trial ERP estimates in a loop 
single_trial_erps = np.zeros((time_samples, ntrials))
for trial in range(ntrials):
    single_trial_erps[:, trial] = eegdata[:,:,trial] @ component1_weights

plt.figure()
plt.plot(time,single_trial_erps)
plt.savefig('Simulated_all_trials_component1.png')
plt.close('all')

# Or use a shortcut with np.einsum() to compute the single-trial ERP estimates
single_trial_erps2 = np.einsum('ijk,j->ik', eegdata, component1_weights)
# plt.figure()
# plt.plot(time,single_trial_erps2)
# plt.savefig('Simulated_all_trials_component1_alt.png')
# plt.close('all')