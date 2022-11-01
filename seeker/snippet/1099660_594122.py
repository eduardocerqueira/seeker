#date: 2022-11-01T17:02:43Z
#url: https://api.github.com/gists/68e44f5ef2ff77dd43bebc630e956dfc
#owner: https://api.github.com/users/galenseilis

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

def duration_calc(s):
    minutes = int(s.split(':')[0])
    seconds = int(s.split(':')[1].split('.')[0])
    milliseconds = int(s.split(':')[1].split('.')[1])
    result = minutes + seconds / 60 + milliseconds / 6000
    return result

# Data
df = pd.read_csv('radiator_pressure_test.csv')
df.Time = df.Time.apply(duration_calc).cumsum()
df['logPSI'] = np.log(df.PSI)

plt.plot(df.Time, df.PSI - np.exp(-0.04 * df.Time + 2.86))
plt.xlabel('Time (minutes)')
plt.ylabel('Residual')
plt.tight_layout()
plt.show()
