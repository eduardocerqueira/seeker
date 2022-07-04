#date: 2022-07-04T16:57:34Z
#url: https://api.github.com/gists/026badbff13641013d76f2bb5b7abae5
#owner: https://api.github.com/users/CamStevens15

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.pyplot import figure
from matplotlib.collections import LineCollection

import numpy as np
import pandas as pd

import fastf1 as ff1
from fastf1 import utils
from fastf1.core import Laps
from fastf1 import plotting

from timple.timedelta import strftimedelta


def main(year, grand_prix, session):
    ''' This calls on the fastf1 api and creates a session based on year,
     grand prix, and session(Quali, race, sprint) wanted '''
    # Setup plotting
    plotting.setup_mpl()

    # Enable the cache
    ff1.Cache.enable_cache('/Users/{"ADD YOUR PATH}/Documents/Coding/F1_2022_race_analytics/cache') 

    # Get rid of some pandas warnings that are not relevant for us at the moment
    pd.options.mode.chained_assignment = None 

    # we only want support for timedelta plotting in this example
    ff1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme=None, misc_mpl_mods=False)

    #This loads the session for the Grand prix session of interest

    session = ff1.get_session(year, grand_prix, session)
    
    session.load() 

    return(session)