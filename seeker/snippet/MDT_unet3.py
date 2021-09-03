#date: 2021-09-03T17:15:27Z
#url: https://api.github.com/gists/a63b532387bf1ec5b0790072c702a106
#owner: https://api.github.com/users/kriz17

def decoder_block_v2(input_decoder, skip_input, num_filters, dropout_=0.1):
  u = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_decoder)
  skip_input = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(skip_input)
  skip_input = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(skip_input)
  skip_input = tf.keras.layers.Dropout(dropout_)(skip_input)
  skip_input = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(skip_input)
  u = tf.keras.layers.concatenate([u, skip_input])
  c = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u)
  c = tf.keras.layers.Dropout(dropout_)(c)
  c = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c)
  return c

#Model Building
def get_unet3(dropout_rate=0.1):
  inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
  c1, e1 = encoder_block(inputs, 64, dropout_rate)
  c2, e2 = encoder_block(e1, 128, dropout_rate)
  c3, e3 = encoder_block(e2, 256, dropout_rate)
  c4, e4 = encoder_block(e3, 512, dropout_rate)
  t = transition_block(e4, 1024, dropout_rate)
  d4 = decoder_block_v2(t, c4, 512, dropout_rate)
  d3 = decoder_block_v2(d4, c3, 256, dropout_rate)
  d2 = decoder_block_v2(d3, c2, 128, dropout_rate)
  d1 = decoder_block_v2(d2, c1, 64, dropout_rate)
  outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(d1)
  model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
  return model

unet3 = get_unet3(dropout_rate=0.1)
unet3.summary()