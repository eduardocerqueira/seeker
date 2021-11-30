#date: 2021-11-30T17:16:08Z
#url: https://api.github.com/gists/12b3bae2790665cc3baaaff91b0cce51
#owner: https://api.github.com/users/rahulremanan

def display_masked_image(img, mask):  
  if len(img.shape)==3:
    img   = img[...,0]
  dim   = img.shape
  img   = cv2.resize(img, (dim[0], dim[1]))
  mask  = cv2.resize(mask, (dim[0], dim[1]))
    
  plt.figure(figsize=(15, 15))
  plt.subplot(1, 3, 1)
  plt.imshow(img,cmap='gray')
  plt.title('Image') 
  plt.axis('off')
   
  plt.subplot(1, 3, 2)
  plt.imshow(mask, cmap='inferno')
  plt.title('Mask')
  plt.axis('off')
 
  plt.subplot(1, 3, 3)
  plt.imshow(img)
  plt.imshow(mask, alpha=0.4, cmap='inferno')
  plt.title('Image + Mask')
  plt.axis('off')
  plt.tight_layout()
  plt.show()