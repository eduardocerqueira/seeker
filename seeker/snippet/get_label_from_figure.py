#date: 2023-10-27T16:35:13Z
#url: https://api.github.com/gists/11f3a0dffd53bfced1376fd0ba45c133
#owner: https://api.github.com/users/josuem

# Define previusly fig = plt.figure()
labels = []

# Iterate through the axes and plots to extract labels
for ax in fig.get_axes():
    for line in ax.lines:
        label = line.get_label()
        labels.append(label)

print(labels)