#date: 2021-09-24T17:12:32Z
#url: https://api.github.com/gists/ec0620398930410e8189002d1ae6d626
#owner: https://api.github.com/users/akshayonly

######################
# Libraries
######################

import streamlit as st
import base64
import json
from datetime import datetime
import sys
import io
from secrets import *

import urllib 
import requests
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd