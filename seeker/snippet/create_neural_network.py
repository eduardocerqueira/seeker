#date: 2023-01-05T16:52:07Z
#url: https://api.github.com/gists/07565e848f941c9c729b960ba63a7cbd
#owner: https://api.github.com/users/JavierMtz5

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def create_nn(self):
    model = Sequential()

    model.add(Dense(64, activation='relu', input_dim=self.state_size))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(self.action_size, activation='linear'))

    model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

    return model