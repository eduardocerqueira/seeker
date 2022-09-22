#date: 2022-09-22T17:18:36Z
#url: https://api.github.com/gists/50934e758d89f50c2c16b07633849862
#owner: https://api.github.com/users/elieclnk

plt.rcParams["figure.figsize"] = [20,12]
from scipy.signal import welch
freqs, psd = welch(df_train, 
                   fs=512, 
                   axis=0)
psd = pd.DataFrame(psd, index = freqs, columns = df_train.columns)
psd.head()
sns.lineplot(data=psd)
plt.xlabel('Hz')
plt.ylabel('Power spectral density')