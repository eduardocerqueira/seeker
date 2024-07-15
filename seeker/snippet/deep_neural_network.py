#date: 2024-07-15T16:52:59Z
#url: https://api.github.com/gists/fa201f1e481c48b28cc879ce362cd2e4
#owner: https://api.github.com/users/kodejuice

import numpy as np
import os
import json


def sigmoid(x):
  x = np.clip(x, -709, 709)  # Clip input to avoid overflow
  return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
  return sigmoid(x) * (1 - sigmoid(x))


def ReLu(v):
  return np.maximum(0, v)


def d_ReLu(x):
  return np.where(x > 0, 1, 0)


def leaky_ReLu(x, alpha=0.01):
  v = np.maximum(alpha * x, x)
  v = np.clip(v, 1e-15, 1 - 1e-15)
  return v


def d_leaky_ReLu(x, alpha=0.01):
  return np.where(x > 0, 1, alpha)


def tanh(x):
  return np.tanh(x)


def d_tanh(x):
  return 1 - np.tanh(x) ** 2


def linear(x):
  x = np.clip(x, 1e-15, 1 - 1e-15)
  return x


def d_linear(x):
  return 1


def softmax(x, T=1):
  clip_value = 10.0
  x = x - x.max(axis=0)
  x = np.clip(x, -clip_value, clip_value)
  exp_xrel = np.exp(x / T)
  return exp_xrel / exp_xrel.sum(axis=0)


class BatchNormLayer:
  def __init__(self, size, epsilon=1e-5, momentum=0.9):
    self.epsilon = epsilon
    self.momentum = momentum
    self.size = size
    self.gamma = np.ones((size, 1))
    self.beta = np.zeros((size, 1))
    self.running_mean = np.zeros((size, 1))
    self.running_var = np.ones((size, 1))

  def forward(self, Z, training=True):
    if training:
      self.Z = Z
      self.mu = np.mean(Z, axis=1, keepdims=True)
      self.var = np.var(Z, axis=1, keepdims=True)
      self.Z_norm = (Z - self.mu) / np.sqrt(self.var + self.epsilon)
      self.Z_out = self.gamma * self.Z_norm + self.beta

      # Update running mean and variance
      self.running_mean = self.momentum * \
          self.running_mean + (1 - self.momentum) * self.mu
      self.running_var = self.momentum * \
          self.running_var + (1 - self.momentum) * self.var
    else:
      Z_norm = (Z - self.running_mean) / \
          np.sqrt(self.running_var + self.epsilon)
      self.Z_out = self.gamma * Z_norm + self.beta

    # print(f"BatchNorm: {self.Z_out}")
    return self.Z_out

  def backward(self, dZ, learning_rate):
    m = dZ.shape[1]

    dgamma = np.sum(dZ * self.Z_norm, axis=1, keepdims=True)
    dbeta = np.sum(dZ, axis=1, keepdims=True)

    dZ_norm = dZ * self.gamma
    dvar = np.sum(dZ_norm * (self.Z - self.mu) * -0.5 *
                  (self.var + self.epsilon) ** (-1.5), axis=1, keepdims=True)
    dmu = np.sum(dZ_norm * -1 / np.sqrt(self.var + self.epsilon), axis=1,
                 keepdims=True) + dvar * np.mean(-2 * (self.Z - self.mu), axis=1, keepdims=True)
    dZ = dZ_norm / np.sqrt(self.var + self.epsilon) + \
        dvar * 2 * (self.Z - self.mu) / m + dmu / m

    # Update gamma and beta
    self.gamma -= learning_rate * dgamma
    self.beta -= learning_rate * dbeta

    return dZ


class NNLayer:
  def __init__(self, input_size, output_size, activation='relu', network_loss=None, keep_prob=1, batch_norm=False):
    self.input_size = input_size
    self.output_size = output_size
    self.activation = activation
    self.__network_loss = network_loss
    self.keep_prob = keep_prob
    self.batch_norm = batch_norm
    self.init_weights()
    if self.batch_norm:
      self.batch_norm_layer = BatchNormLayer(self.output_size)

  def init_weights(self):
    k = 1.
    if self.activation == 'relu':
      k = 2.

    # initialize weights with random values from normal distribution
    self.W = np.random.randn(
      self.output_size, self.input_size) * np.sqrt(k / self.input_size)
    self.b = np.zeros((self.output_size, 1))

    # initialize weights for momentum
    self.vdW = np.zeros((self.output_size, self.input_size))
    self.vdb = np.zeros((self.output_size, 1))

    # initialize weights for Adam
    self.sdW = np.zeros((self.output_size, self.input_size))
    self.sdb = np.zeros((self.output_size, 1))

  def forward(self, A_prev, training=False):
    self.A_prev = A_prev
    self.Z = np.dot(self.W, A_prev) + self.b

    if self.batch_norm:
      self.Z = self.batch_norm_layer.forward(self.Z, training)

    self.A = self.activation_fn(self.Z)

    if training and self.keep_prob < 1:
      # apply dropout to the activations of the previous layer
      self.A = self.A * np.random.binomial(
          1, self.keep_prob, size=self.A.shape)
      # scale the activations
      self.A = self.A / self.keep_prob

    return self.A

  def gradient_descent_update(self, dW, db, learning_rate, L2_reg=0, beta=0.9, beta2=0.999, train_iteration=1, optimization='gd'):
    eps = 1e-8
    if optimization == 'gd':
      self.W -= learning_rate * (dW + L2_reg * self.W)
      self.b -= learning_rate * db
    elif optimization == 'adam':
      self.vdW = beta * self.vdW + (1 - beta) * dW
      self.vdb = beta * self.vdb + (1 - beta) * db
      self.sdW = beta2 * self.sdW + (1 - beta2) * dW ** 2
      self.sdb = beta2 * self.sdb + (1 - beta2) * db ** 2
      # bias correction
      vdW = self.vdW / (1 - beta ** train_iteration)
      vdb = self.vdb / (1 - beta ** train_iteration)
      sdW = self.sdW / (1 - beta2 ** train_iteration)
      sdb = self.sdb / (1 - beta2 ** train_iteration)
      # update weights
      self.W = self.W - learning_rate * vdW / np.sqrt(sdW + eps)
      self.b = self.b - learning_rate * vdb / np.sqrt(sdb + eps)
    elif optimization == 'rmsprop':
      self.vdW = beta * self.vdW + (1 - beta) * dW ** 2
      self.vdb = beta * self.vdb + (1 - beta) * db ** 2
      # update weights
      self.W = self.W - learning_rate * dW / np.sqrt(self.vdW + eps)
      self.b = self.b - learning_rate * db / np.sqrt(self.vdb + eps)
    elif optimization == 'momentum':
      self.vdW = beta * self.vdW + (1 - beta) * dW
      self.vdb = beta * self.vdb + (1 - beta) * db
      self.W -= learning_rate * self.vdW
      self.b -= learning_rate * self.vdb
    else:
      raise ValueError(f"Unsupported optimization method: {optimization}")

  def backward(self, dA, learning_rate, L2_reg=0, beta1=0.9, beta2=0.999, train_iteration=0, optimization='gd'):
    m = self.A_prev.shape[1]

    if self.activation == 'softmax' or self.__network_loss == 'binary_cross_entropy':
      # we already computed the derivative in the nerual network backward pass method
      dZ = dA
    else:
      dZ = dA * self.activation_fn(self.Z, derivative=True)

    if self.batch_norm:
      dZ = self.batch_norm_layer.backward(dZ, learning_rate)

    dW = 1 / m * np.dot(dZ, self.A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(self.W.T, dZ)

    self.gradient_descent_update(
      dW, db, learning_rate,
      L2_reg=L2_reg,
      beta=beta1,
      beta2=beta2,
      train_iteration=train_iteration + 1,
      optimization=optimization
    )

    return dA_prev

  def activation_fn(self, x, derivative=False):
    a = {
        'relu': [ReLu, d_ReLu],
        'leaky_relu': [leaky_ReLu, d_leaky_ReLu],
        'tanh': [tanh, d_tanh],
        'sigmoid': [sigmoid, d_sigmoid],
        'linear': [linear, d_linear],
        'softmax': [softmax, None],
    }
    return a[self.activation][derivative](x)


class Layer:
  def __init__(self, neurons: int, activation='relu', keep_prob=1.0, batch_norm=False):
    self.neurons = neurons
    self.activation = activation
    self.keep_prob = keep_prob
    self.batch_norm = batch_norm


class OutputLayer(Layer):
  def __init__(self, classes: int, activation='linear', keep_prob=1.0, batch_norm=False):
    super().__init__(classes, activation, keep_prob, batch_norm)


def connect_layers(layers: list[Layer], loss=None):
  assert len(layers) > 1, "At least 2 layers are required"
  nn_layers = [
    NNLayer(input_size=layers[0].neurons, output_size=layers[1].neurons,
            activation=layers[1].activation, keep_prob=layers[0].keep_prob),
  ]
  for i in range(1, len(layers) - 1):
    nn_layers.append(
      NNLayer(
        input_size=layers[i].neurons,
        output_size=layers[i + 1].neurons,
        activation=layers[i + 1].activation,
        network_loss=loss,
        keep_prob=layers[i].keep_prob,
        batch_norm=layers[i].batch_norm
      )
    )
  return nn_layers


class DeepNeuralNetwork:
  def __init__(self, layers: list[Layer], loss='mse', L2_reg=0.0, beta1=0.9, beta2=0.999, optimization='gd', model_file=None):
    self.loss_fn = loss
    self.layers = connect_layers(layers, loss)
    self.assertions()
    self.training = False
    self.L2_reg = L2_reg
    self.beta1 = beta1
    self.beta2 = beta2
    self.optimization = optimization
    self.model_file_name = model_file or 'model_weights.json'
    self.load_weights_from_file()

  def assertions(self):
    if self.loss_fn == 'binary_cross_entropy':
      assert self.layers[-1].activation == 'sigmoid', \
          'Last layer must be sigmoid for binary cross entropy'
      assert self.layers[-1].output_size == 1, \
          'Last layer must have 1 neuron for binary cross entropy'

    for layer in self.layers:
      assert layer.activation in [
          'relu',
          'leaky_relu',
          'tanh',
          'sigmoid',
          'linear',
          'softmax',
      ], \
          f"Unsupported activation function: '{layer.activation}'"

    assert self.loss_fn in ['cross_entropy', 'mse', 'binary_cross_entropy'], \
        f"Unsupported loss function: '{self.loss_fn}'"

  def cost(self, X, Y):
    """cost over all examples in a batch"""
    A_L = self.full_forward_pass(X)
    Y = np.array(Y).T

    assert A_L.shape == Y.shape, \
        "Invalid shapes, A_L: %s, Y: %s" % (A_L.shape, Y.shape)

    cost = 0
    if self.loss_fn == 'mse':
      cost = np.mean(np.mean(np.square(A_L - Y), axis=0))
    elif self.loss_fn == 'cross_entropy':
      A_L = np.clip(A_L, 1e-15, 1 - 1e-15)
      cost = np.mean(-np.sum(Y * np.log(A_L), axis=0))
    elif self.loss_fn == 'binary_cross_entropy':
      A_L = np.clip(A_L, 1e-15, 1 - 1e-15)
      cost = np.mean(-np.sum(Y * np.log(A_L) + (1 - Y)
                     * np.log(1 - A_L), axis=0))

    # add L2 regularization
    if self.L2_reg > 0 and self.optimization in ['gd']:
      l2_reg = 0
      for layer in self.layers:
        l2_reg += np.sum(np.square(layer.W))
      cost += (self.L2_reg / 2) * l2_reg

    return cost

  def predict(self, X):
    return self.single_forward_pass(X)

  def single_forward_pass(self, X):
    """Foward pass for a single input"""
    X = np.array(X).reshape((self.layers[0].input_size, 1))
    A = X
    for layer in self.layers:
      A = layer.forward(A)
    return A

  def full_forward_pass(self, X):
    """Foward pass for a batch of inputs"""
    # X_T = np.array(X).T  # make all examples be arranged in a column
    # """
    # X_T = [
    #   [example1_a, example2_a, ..., exampleN_a],
    #   [example1_b, example2_b, ..., exampleN_b],
    #   [example1_c, example2_c, ..., exampleN_c],
    # ]
    # """
    X = np.array(X).T
    A = X
    for layer in self.layers:
      A = layer.forward(A, self.training)
    return A

  def backward_pass(self, Y, learning_rate, iteration=0):
    # we must have run a foward pass before calling this method

    Y = np.array(Y)
    Y_T = Y.T  # reshape training labels to be arranged in a column
    A_L = self.layers[-1].A

    if self.layers[-1].activation == 'softmax' and self.loss_fn == 'cross_entropy':
      dA = A_L - Y_T
    elif self.layers[-1].activation == 'sigmoid' and self.loss_fn == 'binary_cross_entropy':
      dA = A_L - Y_T
    else:
      assert self.loss_fn == 'mse', 'Expected mse loss'
      assert self.layers[-1].activation != 'softmax', 'Use a different activation function other than softmax here'

      dA = 2 * (A_L - Y_T) * \
          self.layers[-1].activation_fn(self.layers[-1].Z, derivative=True)

    for layer in reversed(self.layers):
      dA = layer.backward(
        dA, learning_rate,
        L2_reg=self.L2_reg,
        beta1=self.beta1,
        beta2=self.beta2,
        train_iteration=iteration,
        optimization=self.optimization
      )

  def train(self, X, Y, epochs=900000, initial_learning_rate=0.01, batch_size=64, decay_rate=0.0001, generate_dataset_fn=None, periodic_callback=None):
    print('Initial cost:', self.cost(X, Y))
    if periodic_callback:
      periodic_callback()
      print('')

    if any(l.keep_prob < 1 for l in self.layers):
      print('Applying Dropout to some layers')
    if self.L2_reg > 0 and self.optimization in ['gd']:
      print('Applying L2 regularization')

    for i in range(1, epochs):
      # decay learning rate
      learning_rate = initial_learning_rate / (1 + decay_rate * i)

      # Mini-batch gradient descent
      for j in range(0, len(X), batch_size):
        X_batch = X[j:j + batch_size]
        Y_batch = Y[j:j + batch_size]

        self.training = True
        self.full_forward_pass(X_batch)
        self.backward_pass(Y_batch, learning_rate, iteration=j)
        self.training = False

      if i % 10 == 0:
        self.output_weights_to_file()
        loss = self.cost(X, Y)
        print(f'Epoch {i}, Loss: {loss:.6f}, LR: {learning_rate:.6f}')

      if i % 70 == 0:
        if periodic_callback:
          periodic_callback()

      if i % 100 == 0:
        if generate_dataset_fn:
          X, Y = generate_dataset_fn()
        else:
          # shuffle dataset
          XY = list(zip(X, Y))
          np.random.shuffle(XY)
          X, Y = zip(*XY)

    print('Final cost:', self.cost(X, Y))

  def nn_layers_params(self):
    s = ''
    for layer in self.layers:
      s += f'({layer.input_size}x{layer.output_size}, {layer.activation}) -> '
    return s

  def output_weights_to_file(self):
    layers_params = self.nn_layers_params()
    weights = {'network_params_hash': layers_params, 'weights': []}
    model_file_name = self.model_file_name.replace('.json', '')

    with open(f'{model_file_name}.json', 'w') as f:
      for i, layer in enumerate(self.layers):
        params = {
          'W': layer.W.tolist(),
          'b': layer.b.tolist(),
        }
        if layer.batch_norm:
          params['batch_norm_params'] = {
            'gamma': layer.batch_norm_layer.gamma.tolist(),
            'beta': layer.batch_norm_layer.beta.tolist(),
            'running_mean': layer.batch_norm_layer.running_mean.tolist(),
            'running_var': layer.batch_norm_layer.running_var.tolist(),
          }

        weights['weights'] += [params]
      f.write(json.dumps(weights, indent=1))

  def load_weights_from_file(self):
    model_file_name = self.model_file_name.replace('.json', '')
    if os.path.exists(f'{model_file_name}.json'):
      print('Loading weights from file...')
    else:
      return

    with open(f'{model_file_name}.json', 'r+') as f:
      try:
        model_weights = json.loads(f.read())
      except:
        print('Error: weights file is not valid JSON')
        f.write('{}')
        return

      if 'network_params_hash' not in model_weights:
        print('Error: weights file has no network_params_hash')
        os.rename(f'{model_file_name}.json', f'{model_file_name}_old.json')
        return

      if model_weights['network_params_hash'] != self.nn_layers_params():
        print('Error: weights file and current network layers hash do not match, ignoring')
        # rename old weights file
        os.rename(f'{model_file_name}.json', f'{model_file_name} (old).json')
        return

      weights = model_weights['weights']
      for i, layer in enumerate(self.layers):
        layer.W = np.array(weights[i]['W'])
        layer.b = np.array(weights[i]['b'])
        if layer.batch_norm and 'batch_norm_params' in weights[i]:
          Wi = weights[i]
          layer.batch_norm_layer.gamma = np.array(
            Wi['batch_norm_params']['gamma'])
          layer.batch_norm_layer.beta = np.array(
            Wi['batch_norm_params']['beta'])
          layer.batch_norm_layer.running_mean = np.array(
            Wi['batch_norm_params']['running_mean'])
          layer.batch_norm_layer.running_var = np.array(
            Wi['batch_norm_params']['running_var'])

