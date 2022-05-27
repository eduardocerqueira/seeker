#date: 2022-05-27T17:13:44Z
#url: https://api.github.com/gists/41e7d190a0cebddc13d884d908247217
#owner: https://api.github.com/users/aspose-imaging-cloud-examples

import os
import asposeimagingcloud.models.requests as requests

IMAGE_FILE_NAME= 'example_image.jpeg2000';
IMAGES_FOLDER = 'ExampleImages';
CLOUD_FOLDER = 'CloudImages';
OUTPUT_FOLDER = 'Output';

# Get ClientId and ClientSecret from https://dashboard.aspose.cloud/ 
# or use on-premise version (https://docs.aspose.cloud/imaging/getting-started/how-to-run-docker-container/)
_imaging_api = ImagingApi(client_secret, client_id, 'https://api.aspose.cloud')

def grayscale_from_storage(self):
    """Grayscale an image from cloud storage"""
   
    input_image = os.path.join(IMAGES_FOLDER, IMAGE_FILE_NAME)  
    upload_file_request = requests.UploadFileRequest(os.path.join(CLOUD_FOLDER, IMAGE_FILE_NAME), input_image)
    result = self._imaging_api.upload_file(upload_file_request)
    if result.errors:
        print('Uploading errors count: ' + str(len(result.errors)))
    
    folder = CLOUD_FOLDER  # Input file is saved at the desired folder in the storage
    storage = None  # We are using default Cloud Storage

    request = requests.GrayscaleImageRequest(IMAGE_FILE_NAME, folder, storage)
    updated_image = self._imaging_api.grayscale_image(request)
    
    # Save the image file to output folder   
    new_file_name = "updated_" + IMAGE_FILE_NAME
    path = os.path.abspath(os.path.join(OUTPUT_FOLDER, new_file_name))
    shutil.copy(updated_image, path) 

def create_grayscale_image_from_request(self):
    """Grayscale an image. Image data is passed in a request stream"""
    
    storage = None  # We are using default Cloud Storage
    out_path = None  # Path to updated file (if this is empty, response contains streamed image)  
    input_stream = os.path.join(IMAGES_FOLDER, IMAGE_FILE_NAME)
    
    request = requests.CreateGrayscaledImageRequest(input_stream, out_path, storage)
    updated_image = self._imaging_api.create_grayscaled_image(request)

    # Save the image file to output folder
    new_file_name = "updated_" + IMAGE_FILE_NAME
    path = os.path.abspath(os.path.join(OUTPUT_FOLDER, new_file_name))
    shutil.copy(updated_image, path) 
