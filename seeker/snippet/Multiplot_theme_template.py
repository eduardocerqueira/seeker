#date: 2021-09-22T17:12:52Z
#url: https://api.github.com/gists/b4dfa27fca6f96e9f1fa46cb9759646f
#owner: https://api.github.com/users/Ankitkalauni

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

X = 2* np.random.rand(100,1)
y = 4+3* X * np.random.rand(100,1)
X1 = 21* np.random.rand(100,1)
y1 = 4+11* X * np.random.rand(100,1)
df = np.concatenate([X,y,X1,y1],axis=1)

df = pd.DataFrame(df,columns=['X','y','X1','y1'])


def setup_multiple_plot(data,X,y=None,w=5,h=3,rows=3,dpi=300,HSPACE=0.5,WSPACE=0.5):
    '''
    data: DataFrame

    X: List of the columns to pairplot
    '''



    from itertools import combinations
    import math

    plot = list(combinations(X, 2))

    cols = int(math.ceil(len(plot)/rows))

    #setting plot theme
    plt.rcParams['figure.dpi'] = dpi

    fig = plt.figure(figsize=(w,h), facecolor='#f6f5f5')
    gs = fig.add_gridspec(rows, cols)
    gs.update(wspace=WSPACE, hspace=HSPACE)

    background_color = "#f6f5f5"
    sns.set_palette(['#ffd514','#ff355d'])
    
    #making multiple ax
    ax_dict = {}
    for row in range(rows):
        for col in range(cols):
            ax_dict["ax%s%s" %(row,col)] = fig.add_subplot(gs[row, col])

    locals().update(ax_dict)
    count = 0
    #setting theme for every ax in local()
    for indx,row in enumerate(range(rows)):
        for col in range(cols):
            x = plot[count][0]
            y = plot[count][1]

            count+=1
            locals()['ax' + str(row) + str(col)].tick_params(labelsize=3, width=0.5, length=1.5)

            #comment below 2 lines if you are using regplot
            locals()['ax' + str(row) + str(col)].grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
            locals()['ax' + str(row) + str(col)].grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
       
            for s in ["right", "top"]:
                locals()['ax' + str(row) + str(col)].spines[s].set_visible(False)

            
            locals()['ax' + str(row) + str(col)].set_facecolor(background_color)

            # ax_sns = sns.regplot(x=df[x],y=df[y],ax=locals()['ax' + str(row) + str(col)]
            #                      ,line_kws={'color': '#ff355d','alpha':1})
            
            ax_sns = sns.scatterplot(x=df[x],y=df[y],ax=locals()['ax' + str(row) + str(col)],zorder=2)


            locals()['ax' + str(row) + str(col)].set_facecolor(background_color)
            locals()['ax' + str(row) + str(col)].set_xlabel(x,fontsize=4, weight='bold',)
            locals()['ax' + str(row) + str(col)].set_ylabel(y,fontsize=4, weight='bold')

            
setup_multiple_plot(df,X=df.columns.values)

