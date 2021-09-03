#date: 2021-09-03T17:07:24Z
#url: https://api.github.com/gists/de8df82ec4c902d28d69f3a444e3759a
#owner: https://api.github.com/users/kriz17

def encoder_block(input_encoder, num_filters, dropout_=0.1):
  c = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input_encoder)
  c = tf.keras.layers.Dropout(dropout_)(c)
  c = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c)
  p = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c)
  return c, p

def transition_block(input_transition, num_filters, dropout_=0.1):
  c = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input_transition)
  c = tf.keras.layers.Dropout(dropout_)(c)
  c = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c)
  return c

def decoder_block(input_decoder, skip_input, num_filters, dropout_=0.1):
  u = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_decoder)
  u = tf.keras.layers.concatenate([u, skip_input])
  c = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u)
  c = tf.keras.layers.Dropout(dropout_)(c)
  c = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c)
  return c

#Model Building
def get_unet():  
  inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
  c1, e1 = encoder_block(inputs, 64, 0.1)
  c2, e2 = encoder_block(e1, 128, 0.1)
  c3, e3 = encoder_block(e2, 256, 0.1)
  t = transition_block(e3, 512, 0.1)
  d3 = decoder_block(t, c3, 256, 0.1)
  d2 = decoder_block(d3, c2, 128, 0.1)
  d1 = decoder_block(d2, c1, 64, 0.1)
  outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(d1)
  model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
  return model

unet = get_unet1()
unet.summary()