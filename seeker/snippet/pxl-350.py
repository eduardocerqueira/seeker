#date: 2026-01-01T17:05:19Z
#url: https://api.github.com/gists/62162b127354b3cb2664a94beaccd3b8
#owner: https://api.github.com/users/jameshenry2616-netizen

import numpy as np
import random

# =====================
# Hyperparameters
# =====================
seq_length = 32
embedding_dim = 192
hidden_dim = 487
learning_rate = 0.005
epochs = 100

# =====================
# Training data
# =====================
text = (
    "the quick brown fox jumps over the lazy dog."
    "our sun rises in the east and sets in the west."
    "cats often sleep during the day and play at night."
    "reading books helps improve knowledge and imagination."
    "human brains are very complex and difficult to understand."
)*10

chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
data = np.array([char_to_idx[c] for c in text])

# =====================
# Model parameters
# =====================
np.random.seed(0)

W_embed = np.random.randn(vocab_size, embedding_dim) * 0.01
Wxh = np.random.randn(embedding_dim, hidden_dim) * 0.01
Whh = np.random.randn(hidden_dim, hidden_dim) * 0.01
Why = np.random.randn(hidden_dim, vocab_size) * 0.01

bh = np.zeros(hidden_dim)
by = np.zeros(vocab_size)

# =====================
# Helper functions
# =====================
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# =====================
# Training (with BPTT)
# =====================
for epoch in range(epochs):
    h_prev = np.zeros(hidden_dim)
    loss = 0

    for i in range(0, len(data) - seq_length, seq_length):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = h_prev

        # -------- Forward pass --------
        for t in range(seq_length):
            xs[t] = W_embed[data[i + t]]
            hs[t] = np.tanh(xs[t] @ Wxh + hs[t - 1] @ Whh + bh)
            ys[t] = hs[t] @ Why + by
            ps[t] = softmax(ys[t])
            loss += -np.log(ps[t][data[i + t + 1]])

        # -------- Backprop --------
        dW_embed = np.zeros_like(W_embed)
        dWxh = np.zeros_like(Wxh)
        dWhh = np.zeros_like(Whh)
        dWhy = np.zeros_like(Why)
        dbh = np.zeros_like(bh)
        dby = np.zeros_like(by)

        dh_next = np.zeros(hidden_dim)

        for t in reversed(range(seq_length)):
            dy = ps[t].copy()
            dy[data[i + t + 1]] -= 1

            dWhy += np.outer(hs[t], dy)
            dby += dy

            dh = dy @ Why.T + dh_next
            dh_raw = (1 - hs[t] ** 2) * dh

            dbh += dh_raw
            dWxh += np.outer(xs[t], dh_raw)
            dWhh += np.outer(hs[t - 1], dh_raw)
            dW_embed[data[i + t]] += dh_raw @ Wxh.T

            dh_next = dh_raw @ Whh.T

        # -------- Update --------
        for param, dparam in zip(
            [W_embed, Wxh, Whh, Why, bh, by],
            [dW_embed, dWxh, dWhh, dWhy, dbh, dby],
        ):
            param -= learning_rate * np.clip(dparam, -5, 5)

        h_prev = hs[seq_length - 1]

randomi = random.randint(12, 48)

# =====================
# Text generation
# =====================
def generate(prompt, length=randomi):
    h = np.zeros(hidden_dim)
    result = list(prompt)

    for c in prompt[:-1]:
        x = W_embed[char_to_idx.get(c, 0)]
        h = np.tanh(x @ Wxh + h @ Whh + bh)

    idx = char_to_idx.get(prompt[-1], 0)

    for _ in range(length):
        x = W_embed[idx]
        h = np.tanh(x @ Wxh + h @ Whh + bh)
        y = h @ Why + by
        p = softmax(y)
        idx = np.random.choice(len(p), p=p)
        result.append(idx_to_char[idx])

    return "".join(result)

# =====================
# Interactive loop
# =====================
print(vocab_size)
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    print("XB:", generate(user_input))
