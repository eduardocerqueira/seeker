#date: 2021-08-31T02:25:51Z
#url: https://api.github.com/gists/e66627035cc8fa6ab7f95c8779a741e7
#owner: https://api.github.com/users/gbrfilipe

import pandas as pd

url = 'https://drive.google.com/uc?export=download&id=1rG-zbqVJLQbs8q9CurVDwmJmN3w4h5hK'

header_list=["id","data_atendimento","idade","sexo","mun_residencia","classificacao","comorbidades","situacao_pct","data_obito","data_confirmacao_exame"]
df = pd.read_csv (url, sep=';', header=0, encoding='latin-1', names=header_list)