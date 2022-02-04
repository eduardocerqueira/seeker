#date: 2022-02-04T16:53:33Z
#url: https://api.github.com/gists/f172fde8da08a9326966e25fe896b45f
#owner: https://api.github.com/users/bastings

# Copyright 2022 Google LLC.
# SPDX-License-Identifier: Apache-2.0

class BiLSTMClassifier(nn.Module):
  hidden_size: int
  embedding_size: int
  vocab_size: int
  output_size: int

  @nn.compact
  def __call__(self, inputs, lengths):
    """Embeds and encodes the inputs, and then predicts."""
    embedded = nn.Embedder(
        self.vocab_size, 
        features=self.embedding_size, 
        name='embedder')(
            inputs)
    _, (forward_final, backward_final) = BiLSTM(
        self.hidden_size, 
        name='bilstm')(
            embedded, lengths)
    forward_output = nn.Dense(
        self.output_size, use_bias=False, name='output_layer_fwd')(
            forward_final)
    backward_output = nn.Dense(
        self.output_size, use_bias=False, name='output_layer_bwd')(
            backward_final)
    return forward_output + backward_output  # Logits.