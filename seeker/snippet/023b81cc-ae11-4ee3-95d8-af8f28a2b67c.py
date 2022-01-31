#date: 2022-01-31T17:05:06Z
#url: https://api.github.com/gists/ad0458ba15dfc18534a12722da3913f2
#owner: https://api.github.com/users/ndemir

denorm = lambda x: ((x + 1) / 2).clamp(0, 1)
def show_batch(loader, do_denorm=False):
    fig, ax = plt.subplots(figsize=(24, 12))
    inputs, classes = next(iter(loader))
    if do_denorm:
        inputs = denorm(inputs)
    out = make_grid(inputs, nrow=8).permute(1, 2, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(out)
    
show_batch(train_loader)
