#date: 2025-01-01T17:09:31Z
#url: https://api.github.com/gists/9fb1003f4f891ae9f2cade4b20ddcccf
#owner: https://api.github.com/users/PieroPaialungaAI

largest_amplitudes = np.flip(np.argsort(Y_abs))[0:5]
top_5_amplitude = Y_abs[largest_amplitudes]
top_5_frequencies = frequencies[largest_amplitudes]
top_5_phases = Y_phase[largest_amplitudes]
fft_features = top_5_amplitude.tolist()+top_5_frequencies.tolist()+top_5_phases.tolist()
print(f'The best 5 amplitudes are {top_5_amplitude}')
print(f'The corresponding 5 frequencies are {top_5_frequencies}')
print(f'The corresponding 5 phases are {top_5_phases}')