#date: 2021-11-30T17:13:10Z
#url: https://api.github.com/gists/3fdd2adf2830843f33e721fd4753c686
#owner: https://api.github.com/users/rahulremanan

#ref: https://www.kaggle.com/inversion/run-length-decoding-quick-start
def rle_decode(mask_rle, mask_size, color=1):
  '''
  mask_rle: run-length as string formated (start length)
  shape: (height, width, channels) of array to return 
  color: color for the mask
  Returns numpy array (mask)
  '''
  s = mask_rle.split()
  starts = list(map(lambda x: int(x) - 1, s[0::2]))
  lengths = list(map(int, s[1::2]))
  ends = [x + y for x, y in zip(starts, lengths)]
  mask = np.full((mask_size[0], mask_size[1]), 0)
  img = mask.reshape((mask_size[0] * mask_size[1]))
  for start, end in zip(starts, ends):
    img[start : end] = color
  return img.reshape(mask_size)

def read_mask(imgID, img_size):
  num_rows = len(df[df.id == imgID])  
  masks = list(map(lambda i: rle_decode(df[df.id == imgID].annotation.iloc[i], img_size), 
              range(num_rows)))
  masks = np.array(masks)
  masks = np.moveaxis(masks, 0, 2)
  return masks

def read_png(filename):
  img   = tf.io.read_file(filename)
  img   = tf.image.decode_png(img, channels=INPUT_CHANNELS)
  img   = tf.expand_dims(img, -1)
  img   = tf.cast(img, tf.float32)
  img   = img / 255.
  img   = tf.reshape(img, [*INPUT_SIZE, INPUT_CHANNELS])
  return img

def resize_masks(masks, shape): 
  if len(masks.shape) == 3:
    print(masks.shape)
    print(shape)
    out_masks = list(map(lambda i: cv2.resize(masks[:,:,i].astype(np.uint8), shape), 
                         range(masks.shape[-1])))
    out_masks = np.array(out_masks).reshape([masks.shape[-1], 
                                             shape[1], shape[0]])
    out_masks = np.moveaxis(out_masks, 0, 2) 
  else:
    out_masks = cv2.resize(masks, shape)
  return np.array(out_masks)