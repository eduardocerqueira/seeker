#date: 2025-05-01T17:01:07Z
#url: https://api.github.com/gists/28cac4479cee841b9241e9a2e6f21946
#owner: https://api.github.com/users/SMS-Uni

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to create directories if they do not exist
def create_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

# Function to map the pixel numbers to wavelength ranges below (note to recheck later)
def map_to_wavelength(channel_data, wavelength_min, wavelength_max):
    channel_min = np.min(channel_data)
    channel_max = np.max(channel_data)
    # Normalize the channel data to the wavelength range (important step)
    normalized_data = (channel_data - channel_min) / (channel_max - channel_min)
    mapped_data = wavelength_min + normalized_data * (wavelength_max - wavelength_min)
    return mapped_data

# Specify the folder containing images
image_folder = "File Namee"  # Update this path with the folder containing images needed to seperate!

# List of image paths in the specified folder
image_paths = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

# Directories for output(i.e. the folders)
output_dirs = ["Wavelength_525-575", "Wavelength_580-620", "Wavelength_635-690", "Combined_Images"]
create_directories(output_dirs)

# Process each image
for image_path in image_paths:
    # Open the image and convert to RGB
    img = Image.open(image_path).convert('RGB')
    M = np.asarray(img)
    
    # Map each channel to its respective wavelength range
    band_525_575 = map_to_wavelength(M[:, :, 0], 525, 575)
    band_580_620 = map_to_wavelength(M[:, :, 1], 580, 620)
    band_635_690 = map_to_wavelength(M[:, :, 2], 635, 690)
    
    # Normalize each band separately
    band_525_575_normalized = (band_525_575 - 525) / (575 - 525)
    band_580_620_normalized = (band_580_620 - 580) / (620 - 580)
    band_635_690_normalized = (band_635_690 - 635) / (690 - 635)
    
    # Combine the normalized bands into a single image
    combined_image = np.stack([band_525_575_normalized, band_580_620_normalized, band_635_690_normalized], axis=-1)
    """ 
    # Save the band images to respective folders
    image_name = os.path.basename(image_path).split('.')[0]
    plt.imsave(f"{output_dirs[0]}/{image_name}_525-575nm.tiff", band_525_575, cmap='gray', vmin=525, vmax=575)
    plt.imsave(f"{output_dirs[1]}/{image_name}_580-620nm.tiff", band_580_620, cmap='gray', vmin=580, vmax=620)
    plt.imsave(f"{output_dirs[2]}/{image_name}_635-690nm.tiff", band_635_690, cmap='gray', vmin=635, vmax=690)
    
    # Save the combined image
    plt.imsave(f"{output_dirs[3]}/{image_name}_combined.tiff", combined_image)
    """
    # Plot the original image, band images, and the combined image
    plt.figure(figsize=(12, 12))

    # Original image
    plt.subplot(3, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    # Band 525-575 nm
    plt.subplot(3, 2, 2)
    plt.imshow(band_525_575, cmap='gray', vmin=525, vmax=575)
    plt.title("Band 525-575 nm")
    plt.axis('off')

    # Band 580-620 nm
    plt.subplot(3, 2, 3)
    plt.imshow(band_580_620, cmap='gray', vmin=580, vmax=620)
    plt.title("Band 580-620 nm")
    plt.axis('off')

    # Band 635-690 nm
    plt.subplot(3, 2, 4)
    plt.imshow(band_635_690, cmap='gray', vmin=635, vmax=690)
    plt.title("Band 635-690 nm")
    plt.axis('off')

    # Combined image
    plt.subplot(3, 2, 5)
    plt.imshow(combined_image)
    plt.title("Combined Image")
    plt.axis('off')

    plt.show()

print("Images have been processed, combined, and saved to respective folders.")
