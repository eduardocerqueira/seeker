#date: 2021-11-02T17:06:03Z
#url: https://api.github.com/gists/116765c935bf5d92f0eb82d11cc189a5
#owner: https://api.github.com/users/VikasOjha666

#Cloning the Tensorflow API models repo.
!git clone https://github.com/tensorflow/models.git

#Navigating to models directory.
%cd models/research
# Compile protos.
!protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
!cp object_detection/packages/tf1/setup.py .
!python -m pip install --use-feature=2020-resolver .

#Performing the testing to check whether we have sucessfully installed the TF object detection API.
!python object_detection/builders/model_builder_tf1_test.py