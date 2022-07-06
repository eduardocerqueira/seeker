#date: 2022-07-06T16:59:18Z
#url: https://api.github.com/gists/2340845a7e080065454079d733c037eb
#owner: https://api.github.com/users/alexsandro-matias

import glob
import pandas as pd
arquivos = glob.glob('*.csv')

lista_csv = []

for x in arquivos:
    lista_csv.append(pd.read_csv(x, encoding='latin-1'))

df = pd.concat(lista_csv)