#date: 2022-02-04T16:50:32Z
#url: https://api.github.com/gists/710911bbf264690a91bc79923ad668bf
#owner: https://api.github.com/users/bastings

# Copyright 2022 Google LLC.
# SPDX-License-Identifier: Apache-2.0

class LSTM(nn.Module):
  """A simple unidirectional LSTM."""

  @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1, out_axes=1,
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x):
    return MyLSTMCell(name='cell')(carry, x)

  @staticmethod
  def initialize_carry(batch_dims, hidden_size):
    return MyLSTMCell.initialize_carry(
        jax.random.PRNGKey(0), batch_dims, hidden_size)