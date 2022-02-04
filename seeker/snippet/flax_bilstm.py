#date: 2022-02-04T16:52:00Z
#url: https://api.github.com/gists/83fda65c5262d466465534c231f7f65d
#owner: https://api.github.com/users/bastings

# Copyright 2022 Google LLC.
# SPDX-License-Identifier: Apache-2.0

class BiLSTM(nn.Module):
  """A simple bi-directional LSTM."""
  hidden_size: int

  @nn.compact
  def __call__(self, inputs, lengths):
    batch_size = inputs.shape[0]

    # Forward LSTM.
    initial_state = LSTM.initialize_carry((batch_size,), self.hidden_size)
    _, forward_outputs = LSTM(name='lstm_fwd')(initial_state, inputs)
    forward_final = forward_outputs[jnp.arange(inputs.shape[0]), lengths - 1]

    # Backward LSTM.
    reversed_inputs = flip_sequences(inputs, lengths)
    initial_state = LSTM.initialize_carry((batch_size,), self.hidden_size)
    _, backward_outputs = LSTM(name='lstm_bwd')(initial_state, reversed_inputs)
    backward_final = backward_outputs[jnp.arange(inputs.shape[0]), lengths - 1]

    # Concatenate the forward and backward representations.
    # `outputs` is shaped [B, T, 2*D] and contains all (h) vectors across time.
    backward_outputs = flip_sequences(backward_outputs, lengths)
    outputs = jnp.concatenate([forward_outputs, backward_outputs], -1)

    return outputs, (forward_final, backward_final)