#date: 2023-05-17T16:47:18Z
#url: https://api.github.com/gists/ef64cf02a9264fd16f0687cb0b414af0
#owner: https://api.github.com/users/niklasravnsborg

# Influenced by:
# - https://stackoverflow.com/questions/59458801/how-to-sort-dicom-slices-in-correct-order
# - https://medium.com/@vivek8981/dicom-to-jpg-and-extract-all-patients-information-using-python-5e6dd1f1a07d

import glob
import pydicom as dicom
import os
import cv2
import sys
import numpy as np

def main():
  input_folder = sys.argv[1]
  output_folder = "converted"

  for file in glob.glob('{}/**/*.dcm'.format(input_folder), recursive=True):
    # Load the DICOM file
    ds = dicom.dcmread(file)

    # Uncomment to print all metadata
    # print(ds)

    number = getattr(ds, 'InstanceNumber')
    study = getattr(ds, 'SeriesDescription')
    
    folder_name = os.path.join(output_folder, study)
    
    # Create folder for export if it doesn't exist
    if not os.path.exists(folder_name):
      os.makedirs(folder_name)

    # Convert to float to avoid overflow or underflow losses
    image_2d = ds.pixel_array.astype(float)

    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 256
    
    file_name = os.path.join(folder_name, "{}.jpg".format(str(number)))
    cv2.imwrite(file_name, image_2d_scaled)

    if number % 50 == 0:
      print('{} images converted'.format(number))

if __name__ == "__main__":
  main()
