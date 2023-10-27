#date: 2023-10-27T16:52:41Z
#url: https://api.github.com/gists/712b7db51f0d7420257418cf6b5a4873
#owner: https://api.github.com/users/Avinash07Nayak

print("\033[1mDecoder Class sample outputs\033[0m\n")

#Initialise decoder
decoder = Decoder(vocab_size=vocab_size, embedding_dim=300, emb_matrix=emb_mat, lstm_units=128, att_units=128)

#Generate output from decoder
all_outputs = decoder(example_target_batch, sample_output, sample_hidden_output, sample_cell_output)

print('Shape of decoder input (batch_size,sequence length):', example_target_batch.shape)
print('Shape of decoder output (batch_size,sequence length-1,vocab_size):', all_outputs.shape)