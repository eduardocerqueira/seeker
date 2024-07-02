#date: 2024-07-02T16:54:07Z
#url: https://api.github.com/gists/aca59e087cd23ed829bd3a8aff61fc23
#owner: https://api.github.com/users/vm

import numpy as np
import torch
import torch.nn.functional as F

# Pre-determined token embeddings for tokens in the example sentence
# These embeddings could be pre-trained using word2vec, GloVe, etc.
token_embeddings = "**********"
    'I': [0.2, 0.1, 0.4],
    'love': [0.3, 0.6, 0.1],
    'dogs': [0.8, 0.7, 0.2],
    'cats': [0.6, 0.5, 0.9],
    'because': [0.1, 0.4, 0.7]
}

# Example sentence
sentence = 'I love dogs'

# Split the sentence into tokens
# In practice, you would use a tokenizer that matches the pre-trained embeddings
tokens = "**********"

# Get the embeddings for the tokens in the sentence
# This would typically be done using a look-up in an embedding matrix
embeddings = "**********"=torch.float32)

# Normalize the embeddings to have unit length to ensure that all vectors have the same scale
# This can help with the stability and performance of the model
# The norm is calculated by taking the square root of the sum of the squared values of the vector
norm = embeddings.norm(
    dim=-1, # along the last dimension
    keepdim=True # keep the same dimensions
)
# Normalize the embeddings by dividing by the norm
normalized_embeddings = embeddings / norm

# We need to create weight matrices for the Query (Q), Key (K), and Value (V) transformations
# These are computed by multiplying the normalized embeddings with the corresponding weight matrices
# - Query (Q): Represents the word we are currently focusing on. It asks "What is the context for this word?"
# - Key (K): Represents the word we are comparing against. It answers "How relevant is this word to the query?"
# - Value (V): Represents the actual content or information of the word that will be weighted by the attention mechanism. It answers "What information do we get from this word?"

# Randomly initialize weight matrices for Q, K, and V
# In a trained model, these weights would be learned during training
W_Q = torch.randn(3, 3, dtype=torch.float32)
W_K = torch.randn(3, 3, dtype=torch.float32)
W_V = torch.randn(3, 3, dtype=torch.float32)

# Compute the Q, K, and V matrices by multiplying the normalized embeddings with the corresponding weight matrices
Q = torch.matmul(normalized_embeddings, W_Q)
K = torch.matmul(normalized_embeddings, W_K)
V = torch.matmul(normalized_embeddings, W_V)

# Compute the dot product of Q and K to get the raw attention scores
# The scores represent the relevance of each token to each other token
similarity_scores = torch.matmul(Q, K.T)

# Apply a softmax function to the scores to get the attention weights
# Softmax normalizes the scores so they are all positive and sum to 1
# -1 means the last dimension, so we apply softmax along the last dimension
attention_weights = F.softmax(similarity_scores, dim=-1)

# Multiply the attention weights by the value matrix V to get the weighted sum
# This gives more importance to tokens that have higher attention weights
attention_output = torch.matmul(attention_weights, V)

normalized_norms = normalized_embeddings.norm(dim=1)
print("Norms of Normalized Embeddings:\n", normalized_norms)

print("Attention Weights:\n", attention_weights)
print("Attention Output:\n", attention_output)print("Attention Output:\n", attention_output)