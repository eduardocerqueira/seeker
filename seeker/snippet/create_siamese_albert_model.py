#date: 2022-06-27T16:56:37Z
#url: https://api.github.com/gists/1b6e2896c83cd1e63eec26066a3453a5
#owner: https://api.github.com/users/JanSchm

# Load ALBERT-base model
albert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/2", trainable=True)


# Siamese ALBERT model
input_word_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32)
input_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32)
input_type_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32)
  
albert_layer = albert_encoder({'input_word_ids': input_word_ids, 'input_mask': input_mask, 'input_type_ids': input_type_ids})['pooled_output']
dense_layer = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', name='dense_vector_reduction')(albert_layer)
out = tf.keras.layers.Dense(64, activation=None, name='out')(dense_layer)
out = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(out)
model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=out, name="embedding_model")
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5), loss=tfa.losses.TripletSemiHardLoss())

