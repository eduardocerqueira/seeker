#date: 2022-08-25T15:08:04Z
#url: https://api.github.com/gists/cfb4dd6492fd9ee17a620e2a66e32254
#owner: https://api.github.com/users/fithisux

from minio import Minio
from minio.error import S3Error
client = Minio(
    "my-minio-server:9000",
    access_key= "**********"
    secret_key= "**********"
    secure=False,
)

found = client.bucket_exists("mybucket")
if not found:
    client.make_bucket("mybucket")
else:
    print("Bucket 'mybucket' already exists")


print(list(client.list_objects("mybucket")))