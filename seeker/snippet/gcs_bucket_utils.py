#date: 2021-11-23T17:10:54Z
#url: https://api.github.com/gists/6608148a03dd634ee006dc0d89e7a865
#owner: https://api.github.com/users/josepsmartinez

import os
import base64
from google.cloud import storage
from google.api_core.exceptions import NotFound
from os import path


class Bucket(object):
    """Load bucket configuration and retrive images.

    Parameters
    ----------
    bucket_name : string
        Name of the bucket containing images.
    project : string
        Name of GCP project where bucket is stored.
    folder : string
        Relative path within bucket that precedes image pathes.
    """
    def __init__(
            self, bucket_name: str = None, project: str = None,
            folder: str = '') -> None:
        client = storage.Client(project=project)
        self.bucket = client.bucket(bucket_name)
        self.folder = folder

    def retrieve_image(self, image_path: str, inbase64: bool = True) -> bytes:
        """Retrieve image from bucket.

        Parameters
        ----------
        image_path : str
            Path to the image inside the bucket.
        inbase64 : bool
            Return image as base64.

        Returns
        -------
        Image in base64 encoding.
        """
        image_path = path.join(self.folder, image_path)
        img_blob = self.bucket.blob(image_path)
        try:
            image = img_blob.download_as_string()
        except NotFound:
            return None
        if inbase64:
            image = base64.encodebytes(image)
        return image

    def save_image(self, image_path: str, image_out: str) -> bool:
        """Retrieve image from bucket.

        Parameters
        ----------
        image_path : str
            Path to the image inside the bucket.
        image_out : str
            Path to save the image.

        Returns
        -------
        bool
            True in case downloaded the image, False otherwise.
        """
        image_path = path.join(self.folder, image_path)
        img_blob = self.bucket.blob(image_path)
        img_blob.download_to_filename(image_out)
        if os.path.exists(image_out):
            return True
        return False
