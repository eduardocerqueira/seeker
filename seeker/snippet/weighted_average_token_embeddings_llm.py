#date: 2023-06-30T16:52:06Z
#url: https://api.github.com/gists/cab7915d4719bdb0b695c98e800bbf23
#owner: https://api.github.com/users/JonathanFly

### Implementing of Prompt Blending for a LLM

import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = "**********"
model = AutoModelWithLMHead.from_pretrained('gpt2-xl', device_map='auto')

# Tokenize the entire prompt
prompt = "I am eating today "
input_ids = "**********"='pt')

# Get the embeddings for the entire prompt
all_embeddings = model.transformer.wte(input_ids)

# List of sequences to average
sequences = ["delicious chow mein", "delicious ice cream", "tasty pizza"]

# List of weights for each sequence
weights = [0.6, 0.3, 0.1]
assert len(sequences) == len(weights), "Weights and sequences must have the same length."

# Tokenize and retrieve the embeddings for the sequences
sequence_embeddings = []
for seq in sequences:
    input_ids_seq = "**********"='pt')
    embeddings_seq = model.transformer.wte(input_ids_seq)
    sequence_embeddings.append(embeddings_seq.mean(dim=1))

# Calculate the weighted average embeddings for the desired sequences
weights_tensor = torch.tensor(weights).view(-1, 1, 1).to(all_embeddings.device)
weighted_embeddings = torch.stack(sequence_embeddings, dim=0) * weights_tensor
average_embedding = weighted_embeddings.sum(dim=0)

# Insert position for the averaged embeddings in the prompt
insert_position = 3

# Concatenate the averaged embeddings with the prompt embeddings at the specified position
modified_embeddings = torch.cat([all_embeddings[:, :insert_position], average_embedding.unsqueeze(1), all_embeddings[:, insert_position:]], dim=1)

# Use the modified embeddings as input
output = model.generate(inputs_embeds=modified_embeddings, do_sample=True, max_length=100)
decoded_output = "**********"

print(decoded_output)
ple=True, max_length=100)
decoded_output = tokenizer.decode(output[0])

print(decoded_output)
