#date: 2021-11-30T17:03:08Z
#url: https://api.github.com/gists/d375866cbd7cd98f409f20c3905d1e6a
#owner: https://api.github.com/users/rahulremanan

ROOT_DIR = '/kaggle/input'
DATA_DIR = f'{ROOT_DIR}/sartorius-cell-instance-segmentation'
TRAIN_CSV = f'{DATA_DIR}/train.csv'

INPUT_HEIGHT              = 520
INPUT_WIDTH               = 704
IMAGE_HEIGHT              = 576
IMAGE_WIDTH               = 704
INPUT_CHANNELS            = 3
IMAGE_CHANNELS            = 3
INPUT_SIZE                = [INPUT_HEIGHT, INPUT_WIDTH]
IMAGE_SIZE                = [IMAGE_HEIGHT, IMAGE_WIDTH]