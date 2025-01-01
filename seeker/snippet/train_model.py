#date: 2025-01-01T16:58:41Z
#url: https://api.github.com/gists/f75b62b0b42bbc273e769fbc8f3b7664
#owner: https://api.github.com/users/Mattis1337

import numpy as np
import torch
from torch import nn
from random import shuffle


def train_rnn(dataset, model, criterion, optimizer, device):
    """
    Training a recurrent neural network
    :param dataset: the dataset containing the white/black moves which shall be used for training
    :param model: a model object instantiated from NeuralNetwork
    :param criterion: loss function
    :param optimizer: the optimizer used for enhancing the training algorithm
    :param device: the device currently used for training
    """
    running_loss = 0
    total_states = 0
    model.zero_grad()

    order = [[i] for i in range(len(dataset))]
    print(f'Training on {len(dataset)} randomly shuffled games!')
    shuffle(order)

    for batch, idx in enumerate(order):
        input_sequence, target_sequence = dataset[idx[0]]
        input_sequence.unsqueeze(-1)

        input_sequence.to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        loss = torch.Tensor([0])  # you can also just simply use ``loss = 0``

        input_sequence.squeeze(1)
        # iterating through all previous moves
        for i in range(input_sequence.size(0)):
            total_states += 1
            output = model(input_sequence[i].to(device))

            l = criterion(output.to(device), target_sequence[i].to(device))
            loss += l

        loss.backward()
        optimizer.step()

        running_loss += loss

        if batch % 1000 == 0:
            print(f"average loss: {running_loss / total_states}")
            running_loss = 0
            total_states = 0


# full iterations training
def train_chess_model() -> None:
    """
    This function will train a neural network by creating an instance of
    the neural network class loading the according weights onto it and then
    using the given dataset to train.
    """
    # Getting the number of epochs
    while True:
        try:
            epochs = int(input('Set number of epochs of learning: '))
            break
        except ValueError:
            print(f'Expected epochs to be of type {int}!')

    # setting the number of usable threads
    torch.set_num_threads(8)

    # disabling debugging APIs
    set_debug_apis(False)

    # getting the device which should be used for training
    device = get_training_device()
    outputs = 1863  # last output size
    # defining the output layer
    out_fc = nn.Linear(get_output_shape(chess_topology[1], get_output_shape(chess_topology[0], [12, 8, 8])[0])[0], outputs)
    model = ChessModel(chess_topology[0], chess_topology[1], out_fc)

    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # setting model into training mode
    model.train()

    # we use bitboards to train the AI
    # sequences have to be longer than 1
    # [sequence length, channels, w, h], [sequence length, outputs]
    dataset = [(torch.randn([2, 12, 8, 8], requires_grad=True), torch.randn([2, outputs], requires_grad=True))]
    # changing memory format for RNN models
    model.to(memory_format=torch.channels_last)

    for epoch in range(epochs):
        train_rnn(dataset, model, criterion, optimizer, device)

    print("Done!")


def get_output_shape(model, image_dim):
    """
    Returns torch.Size object to the last layer of a given model by providing it with dummy data and passing
    it through. The model will change the data and its dimension based on its different layers.
    :param model: the model topology
    :param image_dim: the dimensions of the dummy data
    """
    x = torch.rand(image_dim)

    # TODO: this has to be reconfigured (see todo1)
    for module in model:
        if isinstance(module, nn.RNN):
            x = torch.unsqueeze(x, 0)
            _, x = module(x)
        else:
            x = module(x)

    if isinstance(model[0], nn.RNN) or isinstance(model[0], nn.LSTM):
        return [np.shape(x)[1]]

    return np.shape(x)


def get_training_device():
    """
    Looks for available torch devices and returns the highest available
    """

    device = (
        # firstly checks if cuda is available
        "cuda"
        if torch.cuda.is_available()
        # if not the cpu will be used for training
        else "cpu"
    )
    # debug message
    print(f"Currently using {device} device!")

    return device


def set_debug_apis(state: bool):
    """
    Disabling or enabling various debug APIs to enhance regular training without debugging / testing.
    https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    :param state: set APIs to True or False
    """
    torch.autograd.set_detect_anomaly(state)
    torch.autograd.profiler.emit_nvtx(state)
    # The following docs elaborate on the usage of the profiler
    # https://pytorch.org/docs/stable/profiler.html
    torch.autograd.profiler.profile(state)


class ChessModel(nn.Module):
    def __init__(self, conv_seq, fc_seq, out_fc):
        super().__init__()
        self.conv_seq = conv_seq
        self.fc_seq = fc_seq
        self.out_fc = out_fc

    def forward(self, x):
        x = self.conv_seq(x)
        x = torch.flatten(x, 1)

        x = torch.unsqueeze(x, 0)

        for module in self.fc_seq:
            if isinstance(module, nn.RNN):
                x = x.squeeze(-1)
                _, x = module(x)
                x = x[0]
            else:
                x = module(x)

        x = self.out_fc(x)
        return x


chess_topology: list[nn.Sequential] = [
    nn.Sequential(
        nn.Conv2d(12, 32, 5),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4),
        nn.ReLU(),
    ),
    nn.Sequential(
        nn.RNN(64, 576, batch_first=True),
        nn.ReLU(),
        nn.Linear(576, 1152),
        nn.ReLU(),
    )
]

if __name__ == "__main__":
    train_chess_model()
