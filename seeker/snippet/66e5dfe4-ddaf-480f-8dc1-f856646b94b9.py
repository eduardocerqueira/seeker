#date: 2022-01-31T17:05:10Z
#url: https://api.github.com/gists/b3a1043bc55994e009c970b186b7a0cf
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
