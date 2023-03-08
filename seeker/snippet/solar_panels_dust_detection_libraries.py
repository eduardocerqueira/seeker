#date: 2023-03-08T17:11:26Z
#url: https://api.github.com/gists/dd148cdcfa69a053a75b698cd12d22d9
#owner: https://api.github.com/users/suhasmaddali

# Importing the basic libraries to be used in the notebook
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from tqdm import tqdm
from tensorflow.keras.applications import Xception, VGG16, VGG19
from tensorflow.keras.applications import InceptionV3, MobileNet, InceptionV3
import random
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50

import warnings
warnings.filterwarnings('ignore')
print("Is GPU Available: {}".format(tf.config.list_physical_devices('GPU')))