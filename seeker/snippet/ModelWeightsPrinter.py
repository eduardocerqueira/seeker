#date: 2021-08-31T13:03:47Z
#url: https://api.github.com/gists/f1c4261d0152fb5ab9fc82c4ff7f1c85
#owner: https://api.github.com/users/kretes

class ModelWeightsPrinter(Callback):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def on_train_begin(self, logs=None):
        allw = np.hstack([x.flatten() for x in self.model.get_weights()])
        print(allw.shape)
        h = np.histogram(allw, bins=np.linspace(-1, 1, 5))
        print("weights histogram", allw.sum(), h[0], h[1])
        for i, x in enumerate(self.model.get_weights()):
            print("--------------------")
            print(i, x.sum(), x.shape, np.histogram(x, bins=np.linspace(-1, 1, 5)))
        sum_of_w = sum(map(lambda x: x.sum(), self.model.get_weights()))
        print(f"weights train begin {sum_of_w}")

    def on_train_batch_end(self, batch, logs=None):
        sum_of_w = sum(map(lambda x: x.sum(), self.model.get_weights()))
        print()
        print(f"weights {batch=} {sum_of_w}")