#date: 2025-01-01T17:10:55Z
#url: https://api.github.com/gists/3f4aa210621c9c08afe6d718f3f79e8e
#owner: https://api.github.com/users/PieroPaialungaAI

def extract_fft_features( y, x=None,  num_features = 5,max_frequency = 10):
  y= y -np.mean(y)
  # Perform the Fourier Transform
  Y = np.fft.fft(y)
  # Calculate the frequency bins
  if x is None:
    x = np.linspace(0,len(y))
  frequencies = np.fft.fftfreq(len(x), d=(x[1] - x[0]) / (2*np.pi))
  Y_abs = 2*np.abs(Y) / len(x)
  Y_abs[Y_abs < 1e-6] = 0
  relevant_frequencies = np.where((frequencies>0) & (frequencies<max_frequency))
  Y_phase = np.angle(Y)[relevant_frequencies]
  frequencies = frequencies[relevant_frequencies]
  Y_abs = Y_abs[relevant_frequencies]
  largest_amplitudes = np.flip(np.argsort(Y_abs))[0:num_features]
  top_5_amplitude = Y_abs[largest_amplitudes]
  top_5_frequencies = frequencies[largest_amplitudes]
  top_5_phases = Y_phase[largest_amplitudes]
  fft_features = top_5_amplitude.tolist()+top_5_frequencies.tolist()+top_5_phases.tolist()
  amp_keys = ['Amplitude '+str(i) for i in range(1,num_features+1)]
  freq_keys = ['Frequency '+str(i) for i in range(1,num_features+1)]
  phase_keys = ['Phase '+str(i) for i in range(1,num_features+1)]
  fft_keys = amp_keys+freq_keys+phase_keys
  fft_dict = {fft_keys[i]:fft_features[i] for i in range(len(fft_keys))}
  fft_data = pd.DataFrame(fft_features).T
  fft_data.columns = fft_keys
  return fft_dict, fft_data