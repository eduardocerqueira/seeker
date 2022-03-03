#date: 2022-03-03T17:08:50Z
#url: https://api.github.com/gists/53dc2ba761cda7bf231270bc1a874945
#owner: https://api.github.com/users/MiguelOyarzo

#Se selecciona año 2019 y sólo la categoría "Ambos Sexos" de la variable DIM1. Se ordenan los países según la tasa cruda de suicidio anual.
df2019=df[(df["Period"]=="2019")&(df["Dim1"]=="Both sexes")].sort_values("FactValueNumeric", ascending=False)
df2019.head(10)