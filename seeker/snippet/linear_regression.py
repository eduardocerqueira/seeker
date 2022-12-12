#date: 2022-12-12T16:43:34Z
#url: https://api.github.com/gists/c53afb254b104d1d5aa628361b9089d1
#owner: https://api.github.com/users/LiuZhenhai1

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

# confidence interval of 95%
def cal_conf_inter(x=[], y=[], ci=95):

    alpha = 1 - ci / 100
    n = len(x)

    Sxx = np.sum(x**2) - np.sum(x)**2 / n
    Sxy = np.sum(x * y) - np.sum(x)*np.sum(y) / n
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Linefit
    b = Sxy / Sxx
    a = mean_y - b * mean_x

    # Residuals
    def fit(xx):
        return a + b * xx

    residuals = y - fit(x)

    var_res = np.sum(residuals**2) / (n - 2)
    sd_res = np.sqrt(var_res)

    df = n-2                            # degrees of freedom
    tval = stats.t.isf(alpha/2., df) 	# appropriate t value

    def se_fit(x):
        return sd_res * np.sqrt(1. / n + (x - mean_x)**2 / Sxx)
    
    y_err = tval * se_fit(x)
    return  y_err
  
def plotFitLine(x, y, ax, cs, lo1, lo2):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y) # statistic information for linear regression: Linear fitting parameters, significance testing
    y_est = slope * x + intercept
    y_err = cal_conf_inter(x, y, ci=95)
    ax.plot(x, y_est, '-', linewidth=2, c=cs)
    ax.fill_between(x, y_est - y_err, y_est + y_err, color=cs, alpha=0.2)
	
    corr = np.corrcoef(x,y)[0,1] # R
    
    plt.text(lo1, lo2, '$y=%.3fx%+.3f$ \n$R^2=%.3f$ \n$p=%.4f$' % (slope,intercept,(corr*corr),p_value), transform=ax.transAxes, color = cs,size=30)
   
fn = u'E:/temporary/lzh/Permafrost_C/FT_SMMI/ft_time_series.xlsx'
df = pd.read_excel(fn,'Sheet1',index_col=0)
T_on_mean = df['T_on_mean']

fig = plt.figure(figsize=(20,8),dpi=200)
ax1 = fig.add_subplot(111)
l1, = ax1.plot(df.index, T_on_mean, 'o', markersize=13, label='T_on_mean', linewidth=2, c="#E27FAD")
plotFitLine(df.index, T_on_mean, ax1, "#E27FAD", 0.42, 0.3)
plt.show()