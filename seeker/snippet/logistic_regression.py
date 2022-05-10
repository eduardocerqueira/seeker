#date: 2022-05-10T16:59:54Z
#url: https://api.github.com/gists/5b2b9efe1b6cd6c63c62d752af8b7523
#owner: https://api.github.com/users/AlessandroMondin

def logistic_regression(self, X, W, b):
    if self.library == "tf":
        # flatten_X has shape (batch_size, W*H*Channels) --> in our case (64, 32*32*3)
        flatten_X = tf.reshape(X, (-1, W.shape[0]))
        out = self.softmax(tf.matmul(flatten_X, W) + b)
    else:
        flatten_X = X.reshape((-1, W.shape[0]))
        out = self.softmax(torch.matmul(flatten_X,W) + b)
    return out