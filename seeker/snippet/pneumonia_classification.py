#date: 2021-12-30T17:13:20Z
#url: https://api.github.com/gists/5a85a5ee987881729e2f07823884655f
#owner: https://api.github.com/users/AayushGrover101

import glob
import random

import numpy as np
import pandas as pd

# displaying images and manipulating them
from PIL import Image

# progress-bar, matplotlib, and sklearn
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# pytorch
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, resnet34