#date: 2023-12-22T17:04:06Z
#url: https://api.github.com/gists/86c41eb6869b36dc53d5df798e66a040
#owner: https://api.github.com/users/qpwo

# wget https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz

import numpy as np
import torch as torch
import gzip
import pickle
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from abc import ABC, abstractmethod

DEVICE = torch.device("cuda:0")
DTYPE = torch.float16

# load example dataset from http://deeplearning.net/data/mnist/mnist.pkl.gz #
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = [[torch.tensor(x, dtype=DTYPE, device=DEVICE) for x in thing] for thing in pickle.load(f, encoding='bytes')]
    # train_set, valid_set, test_set = torch.tensor(train_set), torch.tensor(valid_set), torch.tensor(test_set)


# establish training data #
train_imgs, train_labels = train_set


# outline layers #
class Layer(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def forward(self, h_in):
        raise NotImplementedError
    @abstractmethod
    def backward_input(self, h_in, h_out, d_hout):
        raise NotImplementedError


class ParamLayer(Layer):
    def __init__(self):
        self.params = dict()

    @abstractmethod
    def backward_param(self, h_in, h_out, d_hout):
        raise NotImplementedError


# initialize weights (glorot_uniform) #
def weight_init(out_size, in_size):
    limit = np.sqrt(6.0 / (in_size + out_size))
    return torch.rand(out_size, in_size, dtype=DTYPE, device=DEVICE) * 2 * limit - limit

def longify(x):
    return torch.tensor(x, dtype=torch.long, device=DEVICE)

# helper fx to set classification = 1, 0 otherwise #
def get_onehot(labels):
    one_hot = torch.zeros((len(labels), 10), dtype=DTYPE, device=DEVICE)
    one_hot[range(len(labels)), longify(labels)] = 1.0
    return one_hot


# Mean Square Error layer #
class MeanSquareError(Layer):
    def forward(self, labels, predict):
        return 2 * ((.5 * (predict - get_onehot(labels)) ** 2).sum(axis=1)).mean(axis=0)

    def backward_input(self, labels, predict):
        return predict - get_onehot(labels)


# Fully Connected Layer #
class FCLayer(ParamLayer):
    def __init__(self, out_size, in_size):
        super(FCLayer, self).__init__()
        self.out_size = out_size
        self.in_size = in_size
        self.params["weight"] = weight_init(out_size, in_size)
        self.params["bias"] = torch.zeros(out_size, dtype=DTYPE, device=DEVICE)  # 1 x out_size

    def forward(self, x):
        return x @ self.params["weight"].T + self.params["bias"]

    def backward_input(self, x, u, de_du):
        return de_du @ self.params["weight"]

    def backward_param(self, x, u, de_du):
        return {"weight": torch.einsum('ij,ik->jk', de_du, x), "bias": torch.sum(de_du, axis=0)}


# Sigmoid activation #
class SigmoidLayer(Layer):
    def forward(self, u):
        return 1 / (1 + torch.exp(-u))

    def backward_input(self, u, O, de_dO):
        return torch.dot(de_dO, (torch.exp(-u) / ((1 + torch.exp(-u)) ** 2)))


# ReLU activation #
class ReLULayer(Layer):
    def forward(self, h_in):
        return torch.clamp(h_in, max=0)
        # return torch.maximum(0, h_in)

    def backward_input(self, h_in, h_out, d_hout):
        d_hout[h_in <= 0] = 0
        return d_hout


# to boost NN accuracy, use ReLU activation function; can substitute with SigmoidLayer() here#
ACT_FUNC = ReLULayer


# Multi-layer perception #
class MLP(object):
    def __init__(self, in_dim, hidden_dims, out_dim):
        self.act_layer = ACT_FUNC()

        dims = [in_dim] + hidden_dims + [out_dim]
        self.fc_layers = []
        for i in range(len(dims) - 1):
            fc_layer = FCLayer(out_size=dims[i + 1], in_size=dims[i])
            self.fc_layers.append(fc_layer)
        self.loss_func = MeanSquareError()

    def forward(self, img_input, img_label):
        x = img_input
        self.hiddens = [x]
        for i in range(len(self.fc_layers)):
            x = self.fc_layers[i].forward(x)
            self.hiddens.append(x)
            if i + 1 < len(self.fc_layers):
                x = self.act_layer.forward(x)
                self.hiddens.append(x)
        logits = x
        loss = self.loss_func.forward(img_label, logits)
        predict = torch.argmax(logits, axis=1)
        accuracy = (predict == img_label).float().mean()
        return loss, accuracy

    def backward(self, img_label):
        grad = self.loss_func.backward_input(img_label, self.hiddens[-1])
        idx = len(self.hiddens) - 1
        self.layer_grads = [None] * len(self.fc_layers)
        for i in range(len(self.fc_layers) - 1, -1, -1):
            assert idx >= 1
            g_param = self.fc_layers[i].backward_param(self.hiddens[idx - 1], self.hiddens[idx], grad)
            self.layer_grads[i] = g_param
            grad = self.fc_layers[i].backward_input(self.hiddens[idx - 1], self.hiddens[idx], grad)
            idx -= 1
            if i > 0:
                grad = self.act_layer.backward_input(self.hiddens[idx - 1], self.hiddens[idx], grad)
                idx -= 1
        assert idx == 0

    def update(self, learning_rate):
        for i in range(len(self.fc_layers)):
            grad_params = self.layer_grads[i]
            params = self.fc_layers[i].params
            params['weight'] -= learning_rate * grad_params['weight']
            params['bias'] -= learning_rate * grad_params['bias']

# initialize MLP #
net = MLP(784, [1024, 1024], 10)


def loop_over_dataset(net, imgs, labels, is_training, batch_size=100):
    loss_list = []
    acc_list = []
    pbar = range(0, imgs.shape[0], batch_size)
    if is_training:
        pbar = tqdm(pbar)
    for i in pbar:
        x = imgs[i: i + batch_size, :]
        y = labels[i: i + batch_size]
        loss, acc = net.forward(x, y)
        if is_training:
            net.backward(y)
            net.update(5e-5)  # learning rate
        loss_list.append(loss.item())
        acc_list.append(acc.item())
        if is_training:
            pbar.set_description('loss: %.4f, acc: %.4f' % (loss, acc))
    if not is_training:
        print('average loss:', np.mean(loss_list))
        print('average accuracy:', np.mean(acc_list))


# train N epochs #
num_epochs = 100
for e in range(num_epochs):
    print('training epoch', e + 1)
    loop_over_dataset(net, train_set[0], train_set[1], is_training=True)
    print('validation')
    loop_over_dataset(net, valid_set[0], valid_set[1], is_training=False)
