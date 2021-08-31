#date: 2021-08-31T13:03:10Z
#url: https://api.github.com/gists/77183205fbe08ff3d29bad42948dab03
#owner: https://api.github.com/users/Ashcom-git

train_log_dir = 'logs/train/'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
batches_per_epoch = len(train_sequence) // (batch_size*3)
min_loss = float('inf')

for epoch in tqdm(range(epochs),position=0):

    loss, max_margin_loss = 0, 0
    pos_gen_data = iter(Dataset.from_generator(PosGenerator(train_sequence, batch_size), output_types=(tuple([tf.int32]*(batch_size)))).repeat(1)) 
    neg_gen_data = iter(Dataset.from_generator(NegGenerator(train_sequence, neg_size, batch_size), output_types=(tuple([tf.int32]*(batch_size)))).repeat(1)) 

    for b in tqdm(range(batches_per_epoch), position=0):
        sen_input = tf.stack(pos_gen_data.get_next())
        neg_input = tf.stack(neg_gen_data.get_next())
   
        batch_loss, batch_max_margin_loss = new_model.train_on_batch([sen_input, neg_input], np.ones((batch_size, 1)))
        loss += batch_loss / batches_per_epoch
        max_margin_loss += batch_max_margin_loss / batches_per_epoch

    with train_summary_writer.as_default():
       tf.summary.scalar('loss', loss, step=epoch)

    if loss < min_loss:
        min_loss = loss
        vectors = new_model.get_layer('asp_emb').weights[0].numpy()
        n_clusters = vectors.shape[0]
        for i in range(n_clusters):
          vector = vectors[i, :]
          top_n_words = emb_reader.get_model().wv.similar_by_vector(vector, topn=50, restrict_vocab=None)
          print('\nAspect {}:'.format(i))
          for word in top_n_words:
            print(word[0]+':'+str(round(word[1],3)),end=' | ')
          print() 

    print('\nEpoch {}'.format(epoch))
    print('Total loss: {}, Maximum margin loss: {}'.format(loss, max_margin_loss))
    print('='*865)
    print()    