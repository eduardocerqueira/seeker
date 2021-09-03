#date: 2021-09-03T17:13:16Z
#url: https://api.github.com/gists/76900b12335bafa56aab2c9751bbca09
#owner: https://api.github.com/users/kriz17

#Model Building

def get_unet2(dropout_rate=0.1):
  inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
  c1, e1 = encoder_block(inputs, 64, dropout_rate)
  c2, e2 = encoder_block(e1, 128, dropout_rate)
  c3, e3 = encoder_block(e2, 256, dropout_rate)
  c4, e4 = encoder_block(e3, 512, dropout_rate)
  t = transition_block(e4, 1024, dropout_rate)
  d4 = decoder_block(t, c4, 512, dropout_rate)
  d3 = decoder_block(d4, c3, 256, dropout_rate)
  d2 = decoder_block(d3, c2, 128, dropout_rate)
  d1 = decoder_block(d2, c1, 64, dropout_rate)
  outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(d1)
  model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
  return model

unet2 = get_unet2(dropout_rate=0.1)
unet2.summary()