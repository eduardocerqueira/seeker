#date: 2022-04-29T17:02:57Z
#url: https://api.github.com/gists/6880a18be286a0f0f217e1d7ba82ba4e
#owner: https://api.github.com/users/DanqingYANG

import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")

# Set up code checking
import os
if not os.path.exists("../input/a.csv"):
    os.symlink("../input/data-for-datavis/a.csv", "../input/a.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex3 import *
print("Setup Complete")

