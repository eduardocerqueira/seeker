#date: 2022-08-31T17:16:28Z
#url: https://api.github.com/gists/007c6b5286c8c2726c7c0bf400700e68
#owner: https://api.github.com/users/eugsim1

import oml
oml.connect(user= "**********"='oracle', host='localhost', port=1521, service_name='orclpdb19', automl=True)
print(oml.isconnected())
print(oml.isconnected(check_automl=True))
print(oml.check_embed())
import numpy
print('numpy version:'+ ' ' + numpy.__version__)
import pandas
print('pandas version:'+ ' '+ pandas.__version__)
import scipy
print('scipy version:'+ ' '+ scipy.__version__)
import matplotlib
print('matplotlib version:'+ ' '+ matplotlib.__version__)
import cx_Oracle
print('cx_Oracle version:'+ ' '+ cx_Oracle.__version__)
import sklearn
print('sklearn version:'+ ' '+ sklearn.__version__)rsion__)