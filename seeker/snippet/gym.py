#date: 2021-09-29T16:45:39Z
#url: https://api.github.com/gists/2483e7b6ebfd27d4315b6890b7ef1808
#owner: https://api.github.com/users/forktheweb

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.layers import GlobalAveragePooling1D, LSTM, Bidirectional

from bad_content import config
from bad_content.utils import show_plot_confusion_matrix, show_classification_report

warnings.filterwarnings("ignore")  # We're outlaws!


def create_embedding_matrix(filepath, word_index, embedding_dim):
    print('Creating embedding matrix from the glove.')
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath, encoding='utf8') as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


def train(classification_report: bool = False, plot_confusion_matrix_report: bool = False) -> None:
    """For better result while training, play https://www.youtube.com/watch?v=_YYmfM2TfUA as loud as possible."""

    df = pd.read_csv('data/bad_content_clean.csv', encoding='utf-8')
    df.head()

    data = df.copy()  # Make a copy of the data.

    print(f'Value Count: {data.spam.value_counts()}')

    # sns.countplot(data['spam'])
    # plt.show()

    X = data['content'].values
    y = data['spam'].values

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Prepare the tokenizer.
    t = Tokenizer()
    t.fit_on_texts(X_train)

    # integer encode the documents
    encoded_train = t.texts_to_sequences(X_train)
    encoded_test = t.texts_to_sequences(X_test)
    print(f'encoded_train[0:2]: {encoded_train[0:2]}')

    # pad documents to a max length of 50 words.
    max_length = 50
    padded_train = pad_sequences(encoded_train, maxlen=max_length, padding='post')
    padded_test = pad_sequences(encoded_test, maxlen=max_length, padding='post')

    print(f'padded_train: {padded_train}')

    vocab_size = len(t.word_index) + 1

    embedding_dim = max_length
    embedding_matrix = create_embedding_matrix(
        f'data/glove.6B/glove.6B.{embedding_dim}d.txt',
        t.word_index,
        embedding_dim
    )

    def my_model():
        # Define the model as Sequential.
        model = Sequential()

        # The model trains for a number of epochs and stops once it is not improving anymore.
        # This is made possible by the early [stopping callback](https://keras.io/api/callbacks/early_stopping/).
        # The model training might run for about 11 or 12 epochs.
        # This varies because of the stochastic[https://machinelearningmastery.com/stochastic-in-machine-learning/]
        # nature of the model and even data splitting.
        model.add(Embedding(vocab_size, embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=False))

        # model.add(Flatten())
        model.add(GlobalAveragePooling1D())
        model.add(Dense(X_train.shape[0] / 4, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(X_train.shape[0] / 6, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(X_train.shape[0] / 8, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(X_train.shape[0] / 10, activation='relu')) 
        model.add(Dropout(0.2))

        model.add(Dense(1, activation='sigmoid'))

        # compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # summarize the model
        print(f'model.summary(): {model.summary()}')

        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        # checkpoint = ModelCheckpoint(
        # 'models/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        #     monitor='val_accuracy',
        #     save_best_only=True,
        #     verbose=1,
        # )

        # fit the model
        model.fit(
            x=padded_train,
            y=y_train,
            epochs=100,
            # batch_size=20,
            validation_data=(padded_test, y_test),
            verbose=1,
            # callbacks=[checkpoint, early_stop],
            callbacks=[early_stop, ],
            use_multiprocessing=True
        )

        return model

    def ltsm_model():
        # LSTM hyperparameters
        n_lstm = 20
        drop_lstm = 0.2

        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
        model.add(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True))
        model.add(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # summarize the model
        print(f'model.summary(): {model.summary()}')

        num_epochs = 30
        early_stop = EarlyStopping(monitor='val_loss', patience=2)
        model.fit(
            padded_train,
            y_train,
            epochs=num_epochs,
            validation_data=(padded_test, y_test),
            callbacks=[early_stop],
            verbose=1,
        )

        return model

    def bi_lstm_model():
        # LSTM hyperparameters
        n_lstm = 20
        drop_lstm = 0.2
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
        model.add(Bidirectional(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True)))
        # model.add(Bidirectional(CuDNNLSTM(
        #     units=n_lstm,
        #     dropout=drop_lstm,
        #     return_sequences=True,
        #     recurrent_activation='sigmoid',
        # )))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # summarize the model
        print(f'model.summary(): {model.summary()}')

        num_epochs = 30
        early_stop = EarlyStopping(monitor='val_loss', patience=2)

        model.fit(
            padded_train, y_train, epochs=num_epochs,
            validation_data=(padded_test, y_test),
            callbacks=[early_stop, ],
            verbose=1,
            use_multiprocessing=True
        )

        return model

    model = bi_lstm_model()

    preds = (model.predict(padded_test) > 0.5).astype("int32")

    if classification_report:
        show_classification_report(y_test, preds)

    if plot_confusion_matrix_report:
        show_plot_confusion_matrix(y_test, preds)

    if not os.path.exists(config.__MODEL_SAVE_PATH):
        os.makedirs(config.__MODEL_SAVE_PATH)

    print(f'Saving model to {config.__MODEL_SAVE_PATH}')

    model.save(config.__MODEL_SAVE_PATH)

    with open(f'{config.__MODEL_SAVE_PATH}/tokenizer.pkl', 'wb') as output:
        pickle.dump(t, output, pickle.HIGHEST_PROTOCOL)
