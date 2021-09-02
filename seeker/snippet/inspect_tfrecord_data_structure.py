#date: 2021-09-02T16:53:11Z
#url: https://api.github.com/gists/a15a7778fd5cf0784f69abddd2d8acc5
#owner: https://api.github.com/users/aswinvk28

filenames = ['/content/drive/MyDrive/tpu-getting-started/tfrecords-jpeg-224x224/train/00-224x224-798.tfrec']
raw_dataset = tf.data.TFRecordDataset(filenames)

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)