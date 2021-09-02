#date: 2021-09-02T17:16:09Z
#url: https://api.github.com/gists/0114fac1230eac94e245f812f314c94b
#owner: https://api.github.com/users/RailanDeivid

#agrupando dados por UF e contando
reclamacoes_por_estado = df.groupby('UF')['UF'].count()\
                                               .sort_values(ascending=False)\
                                               .to_frame(name="QUANTIDADE")\
                                               .reset_index()
#visualizando os dados
reclamacoes_por_estado.head()