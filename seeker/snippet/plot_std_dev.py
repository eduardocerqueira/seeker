#date: 2022-05-16T17:15:38Z
#url: https://api.github.com/gists/3ee387c72b94e6371c9476dd7d53979b
#owner: https://api.github.com/users/theDestI

def plot_std_dev(df):
    """
    Plots the standard deviation of a series.

    :param series: The series to be plotted.

    :return: The plot.
    """
    price = df['Close']
    mean = price.mean()
    std = price.std()

    plt.plot(df.index, (price - mean) / std)
    plt.fill_between(
        df.index,
        (price - mean) / std,
        (price - mean) / std,
        color='blue',
        alpha=0.2,
    )
    plt.show()