#date: 2023-11-07T17:09:22Z
#url: https://api.github.com/gists/db6b98672675456bed39d45077d44179
#owner: https://api.github.com/users/Poiuytrezay1

import os
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
import library.model_util as model_util
from torchvision import transforms

DEVICE_CUDA = torch.device("cuda:0")
IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def load_image(image_path):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    img = np.array(image, np.uint8)
    return img, image.info

def process_images_group(vae, images_group):
    with torch.no_grad():
        # Stack the tensors from the same size group
        img_tensors = torch.stack(images_group, dim=0).to(DEVICE_CUDA)
        
        # Encode and decode the images
        latents = vae.encode(img_tensors).latent_dist.sample()
        
        decoded_images = []
        for i in range(0, 1, 1):
            decoded_images.append(
                vae.decode(latents[i : i + 1] if i > 1 else latents[i].unsqueeze(0)).sample
            )
        decoded_images = torch.cat(decoded_images)

        return decoded_images

def get_image_from_latents(vae, input_dir, output_dir, batch_size=1):
  ### READ IMAGES FROM input_dir ###
  vae.to(DEVICE_CUDA)

  image_files = [file for file in os.listdir(input_dir) if file.endswith(('jpg', 'jpeg', 'png'))]
  size_to_images = defaultdict(list)
  file_names = [] # List to keep track of file names

  for image_file in image_files:
      image_path = os.path.join(input_dir, image_file)
      image, _ = load_image(image_path)
      transformed_image = IMAGE_TRANSFORMS(image)
      size_to_images[transformed_image.shape[1:]].append(transformed_image)
      file_names.append(image_file) # Save the file name

  os.makedirs(output_dir, exist_ok=True)

  for size, images_group in size_to_images.items():
    # Process images in batches
    for i in range(0, len(images_group), batch_size):
      batch = images_group[i:i + batch_size]
      batch_file_names = file_names[i:i + batch_size] # Get the batch file names
      decoded_images = process_images_group(vae, batch)

      # Rescale images from [-1, 1] to [0, 255] and save
      decoded_images = ((decoded_images / 2 + 0.5).clamp(0, 1) * 255).cpu().permute(0, 2, 3, 1).numpy().astype("uint8")
      for j, decoded_image in enumerate(decoded_images):
          original_file_name = batch_file_names[j] # Get the original file name for each image
          original_name_without_extension = os.path.splitext(original_file_name)[0]
          Image.fromarray(decoded_image).save(os.path.join(output_dir, f"{original_name_without_extension}-latents.png")) # Save with the original name and '-latents'

input_dir = "./input_images" # input images
output_dir = "./output_latents" # output to store the decoded latents
vae_path = "./kl-f8-anime2.ckpt"
name_or_path = "" # model path to extract the VAE from (if vae_path not set)

if len(vae_path) == 0:
  # Load model's VAE
  text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(
      False, name_or_path, DEVICE_CUDA, unet_use_linear_projection_in_v2=False
  )
else:
  vae = model_util.load_vae(vae_path, torch.float32)

# Save image decoded latents
get_image_from_latents(vae, input_dir, output_dir)