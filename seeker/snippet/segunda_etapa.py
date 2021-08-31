#date: 2021-08-31T02:44:56Z
#url: https://api.github.com/gists/ec6f70489ca4f2f3758ce23e982dc3bd
#owner: https://api.github.com/users/gbrfilipe

import unicodedata
from datetime import datetime
import xlrd

def serial_to_datetime(x):
        if pd.isnull(x):
                return
        else:
                return datetime(*xlrd.xldate_as_tuple(x, 0))
 
def remove_accents(string: str) -> str:
    normalized = unicodedata.normalize('NFKD', string)
    return ''.join([c for c in normalized if not unicodedata.combining(c)])

df["classificacao"] = df["classificacao"].apply(remove_accents).str.upper()
df["situacao_pct"] = df["situacao_pct"].apply(remove_accents).str.upper()
df["comorbidades"] = df["comorbidades"].fillna('').astype(str).apply(remove_accents).str.upper()
df.drop(df[df['idade'] < 0].index, inplace=True)
df["data_atendimento"] = df["data_atendimento"].apply(serial_to_datetime)
df["data_obito"] = df["data_obito"].apply(serial_to_datetime)
df["mun_residencia"] = df["mun_residencia"].apply(remove_accents).str.upper()
df["data_confirmacao_exame"] = df["data_confirmacao_exame"].apply(serial_to_datetime)
df["sexo"] = df["sexo"].str.upper().replace("FEMININO", "F").replace("MASCULINO", "M").str.lstrip()