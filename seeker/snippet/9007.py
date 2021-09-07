#date: 2021-09-07T16:46:35Z
#url: https://api.github.com/gists/ae2c7dff349359469069185baca1c314
#owner: https://api.github.com/users/BexTuychiev

fig, ax = plt.subplots(figsize=(12, 9))

sns.histplot(
    data=tps_df, x="f6", label="Original data", color="red", alpha=0.3, bins=15
)
sns.histplot(
    data=sample_df, x="f6", label="Sample data", color="green", alpha=0.3, bins=15
)

plt.legend()
plt.show();