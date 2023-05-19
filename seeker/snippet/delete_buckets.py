#date: 2023-05-19T17:05:00Z
#url: https://api.github.com/gists/f0b6d076d6f9098c4892f9b439201701
#owner: https://api.github.com/users/samkeen

import boto3

# This code deletes buckets on AWS S3.
# It is designed to be run on a local machine with credentials for an AWS account that has the
# necessary permissions to delete buckets.
# It will delete the contents of the buckets before deleting the bucket itself.
# It will not attempt to delete a bucket that does not exist.

# BUCKETS is a list of bucket names that should be deleted. If a bucket in the list does not
# exist, it will be skipped.
BUCKETS = [

]

# Get a session based on profile name
def get_boto_session():
    try:
        return boto3.Session(profile_name="")
    except Exception as e:
        print(f"There was an unknown error when creating boto session: {e}")
        exit(1)

# Delete all objects in a bucket
def delete_bucket_contents(bucket_objects: dict):
    if "Contents" not in bucket_objects:
        return
    # For each object, delete it
    for object in bucket_objects["Contents"]:
        print(f"\tDeleting object {object['Key']}")
        s3.delete_object(Bucket=bucket_name, Key=object["Key"])


# Create an S3 client
boto_session = get_boto_session()
s3 = boto_session.client("s3")

# For each bucket, empty it and then delete it
for bucket_name in BUCKETS:
    print(f"Deleting bucket {bucket_name}")
    # check if bucket exists
    try:
        s3.head_bucket(Bucket=bucket_name)
    except Exception as e:
        print(f"Bucket {bucket_name} does not exist")
        continue

    # Get the objects in the bucket
    objects = s3.list_objects(Bucket=bucket_name)

    delete_bucket_contents(objects)

    # Delete the bucket itself
    s3.delete_bucket(Bucket=bucket_name)
    print(f"\n----\n")
