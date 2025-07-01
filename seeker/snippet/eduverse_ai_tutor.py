#date: 2025-07-01T17:09:29Z
#url: https://api.github.com/gists/10b40dc1a3a32134ea7c9bb70076774d
#owner: https://api.github.com/users/B45678-YAHOO

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import string

# Sample educational content
text = """Welcome to EduVerse. Your first mission is to balance a budget for a clean energy startup. 
You must allocate funds across research, operations, and marketing. Type your decision and I will guide you."""

# Preprocess text
text = text.lower().translate(str.maketrans('', '', string.punctuation))
chars = sorted(set(text))
char2idx = {char: idx for idx, char in enumerate(chars)}
idx2char = np.array(chars)

encoded_text = np.array([char2idx[c] for c in text])

seq_length = 50
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    return chunk[:-1], chunk[1:]

dataset = sequences.map(split_input_target).batch(64).shuffle(1000)

vocab_size = len(chars)
embedding_dim = 256
rnn_units = 512

model = Sequential([
    Embedding(vocab_size, embedding_dim),
    LSTM(rnn_units, return_sequences=True),
    Dense(vocab_size)
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(dataset, epochs=10)

def generate_text(model, start_string, num_chars=300):
    input_eval = [char2idx[s] for s in start_string.lower()]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    model.reset_states()
    for _ in range(num_chars):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions[-1:], num_samples=1).numpy()[0][0]

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

print(generate_text(model, start_string="Your mission is"))
