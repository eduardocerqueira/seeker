#date: 2025-01-01T17:11:53Z
#url: https://api.github.com/gists/34931a9d7a6199486027e8321da89785
#owner: https://api.github.com/users/PieroPaialungaAI

x = np.linspace(-8*np.pi, 8*np.pi, 10000)
y = np.sin(x) + 0.4*np.cos(2*x) + 2*np.sin(3.2*x)
extract_fft_features(x=x,y=y,num_features=3)[1]