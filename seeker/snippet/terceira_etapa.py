#date: 2021-08-31T03:02:34Z
#url: https://api.github.com/gists/a7e842bdf6aa6773b3addee504c10cf7
#owner: https://api.github.com/users/gbrfilipe

import sqlalchemy as sq
engine = sq.create_engine("postgresql+psycopg2://USUARIO:SENHA*@HOST:PORTA/postgres", client_encoding='latin1')

df2 = pd.read_sql_table("covid19_al", engine)