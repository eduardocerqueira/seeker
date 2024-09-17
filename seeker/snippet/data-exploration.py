#date: 2024-09-17T17:09:06Z
#url: https://api.github.com/gists/262f97965307559af70c6cdf976ea6a0
#owner: https://api.github.com/users/docsallover

# How many articles per subject?
print(data.groupby(['subject'])['text'].count())
data.groupby(['subject'])['text'].count().plot(kind="bar")
plt.show()

# How many fake and real articles?
print(data.groupby(['target'])['text'].count())
data.groupby(['target'])['text'].count().plot(kind="bar")
plt.show()