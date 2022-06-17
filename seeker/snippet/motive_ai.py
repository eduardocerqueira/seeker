#date: 2022-06-17T17:12:02Z
#url: https://api.github.com/gists/c4893a8ee35e83f99341dc09fdaaa735
#owner: https://api.github.com/users/sumairhw

from google.colab import drive
drive.mount('/content/drive')

%cp /content/drive/MyDrive/motive_ai/* .
!unzip single_file.zip && rm single_file.zip
!unzip public_testset_images.zip && rm public_testset_images.zip 

!python motive_to_yolo.py train_gt.json
!python split_train_val.py train_images/ labels/

%rm -rf sample_data train_images train_gt.json labels

drive.flush_and_unmount()

!git clone https://github.com/ultralytics/yolov5  # clone
%pip install -qr yolov5/requirements.txt  # install