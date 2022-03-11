#date: 2022-03-11T16:52:14Z
#url: https://api.github.com/gists/5c2dd1edb9ed41bbb8a3f1d760f5d654
#owner: https://api.github.com/users/yubin8773

def plot_learning_curve(log_df,
                        loss_name='loss',
                        rolling=False,
                        ylim=(None, None), **kwargs):
    '''
    A simple function for plotting a learning curve of the model

    Args:
        log_df: input pandas Dataframe
        loss_name: name of the loss
        ylim: y-axis limits, Tuples of (bottom, top)
        rolling: Defaults to False. If set to True, plot moving averaged loss graph in the second subplot

    Author: SungJae Lee, Co-author: Yubin Lee
    Last Modified: 2022.03.12
    '''

    # Data from the log.csv
    epochs = np.arange(log_df.epoch.iloc[0] + 1, log_df.epoch.iloc[-1] + 2, 1, dtype=np.uint32)

    plt.style.use('seaborn-whitegrid')
    fig1 = plt.figure(figsize=kwargs.get('fig_size', (8, 4)))

    plt.title('Learning Curves (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    if ylim[0] is not None:
        plt.ylim(bottom=ylim[0])
    if ylim[1] is not None:
        plt.ylim(top=ylim[1])

    plt.plot(epochs, log_df[f'{loss_name}'], '-', label='Training')
    plt.plot(epochs, log_df[f'val_{loss_name}'], '-', label='Validation')
    plt.legend()

    fig1.tight_layout()
    fig1.show()

    if rolling:
        fig2 = plt.figure(figsize=kwargs.get('fig_size', (8, 4)))
        loss_mavg = log_df[f'{loss_name}'].rolling(window=5).mean()
        val_loss_mavg = log_df[f'val_{loss_name}'].rolling(window=5).mean()

        plt.plot(epochs, loss_mavg, '-', label='Training')
        plt.plot(epochs, val_loss_mavg, '-', label='Validation')

        fig2.tight_layout()
        fig2.show()