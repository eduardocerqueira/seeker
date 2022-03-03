#date: 2022-03-03T17:14:37Z
#url: https://api.github.com/gists/90b0fb13dd1c305f4174a0e34aa58d9d
#owner: https://api.github.com/users/MiguelOyarzo

#Agrupación por continentes de la tasa de suicidio, considerando todos los años de la base (2000-2019). Se incorpora el máximo y mínimo. Sólo considera categoría de ambos sexos.
df_cont=dftotal.groupby("ParentLocation")["FactValueNumeric"].agg([min,max,np.mean])
df_cont=df_cont.reset_index().sort_values("mean", ascending=False)
df_cont