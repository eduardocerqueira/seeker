#date: 2022-02-10T17:00:52Z
#url: https://api.github.com/gists/67265c79b6cd6f6aeeadab37c431e05f
#owner: https://api.github.com/users/marshmellow77

from datasets.filesystems import S3FileSystem

s3 = S3FileSystem()  

training_input_path = f's3://{sess.default_bucket()}/{s3_prefix}/data'
train_dataset.save_to_disk(training_input_path, fs=s3)