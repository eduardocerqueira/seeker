#date: 2024-09-16T16:57:20Z
#url: https://api.github.com/gists/05da91b9b34799ff6fd4254cffba7d3e
#owner: https://api.github.com/users/rlank

from google.cloud import storage
from datetime import timedelta
import os

# Set the path to your service account key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/path/to/key/json'

def generate_signed_urls(bucket_name, prefix, expiration_time):
    """
    Generates signed URLs for files in the given bucket and prefix.

    :param bucket_name: Name of the GCS bucket.
    :param prefix: Prefix of the files in the GCS bucket.
    :param expiration_time: Time in minutes for which the signed URL should be valid.
    :return: List of tuples containing the file name and signed URL.
    """
    # Initialize the client
    # This uses the default credentials. Make sure that the GOOGLE_APPLICATION_CREDENTIALS environment variable is set.
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Get blobs (files) with the given prefix
    blobs = bucket.list_blobs(prefix=prefix)

    signed_urls = []
    for blob in blobs:
        # Generate a signed URL for each blob
        url = blob.generate_signed_url(
            expiration=expiration_time,
            version='v4'  # Use V4 signing
        )
        signed_urls.append((blob.name, url))

    return signed_urls
# Usage
bucket_name = 'fuelcast-data'
prefix = 'fuel/rapid-2024-conus/'

# Longest allowable time is one week
exp_time = timedelta(days=7)

signed_urls = generate_signed_urls(bucket_name, prefix, expiration_time=exp_time)

# Print signed URLs
for file_name, url in signed_urls:
    print(f"File: {file_name} - Signed URL: {url}")