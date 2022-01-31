#date: 2022-01-31T17:04:21Z
#url: https://api.github.com/gists/a1c87c99c5369a4ca2331349e7b8ae0f
#owner: https://api.github.com/users/ndemir

def plot(*loss_history):
    keys = loss_history[0].keys()
    for k in keys:
        plt.figure()
        data = []
        for l in loss_history:
            data.extend(l[k])
        seaborn.lineplot(x=range(len(data)), y=data).set_title(k)
plot(loss_history)
plot(val_metrics_history)
